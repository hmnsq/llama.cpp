#నిర్వచించు _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#నిర్వచించు USE_MATH_నిర్వచిస్తుంది // For M_PI on MSVC

#అన్తర్భావ "ggml-version.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-threading.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml.h"

// FIXME: required here for quantization functions
#అన్తర్భావ "ggml-quants.h"

#నిర్వచితమైతే GGML_USE_CPU_HBM
#అన్తర్భావ <hbwmalloc.h>
#అయితే_ముగింపు

#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
#అన్తర్భావ <malloc.h> // using malloc.h with MSC/MINGW
#లేదావేళ !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#అన్తర్భావ <alloca.h>
#అయితే_ముగింపు

#అన్తర్భావ <assert.h>
#అన్తర్భావ <errno.h>
#అన్తర్భావ <time.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <inttypes.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <float.h>
#అన్తర్భావ <limits.h>
#అన్తర్భావ <stdarg.h>
#అన్తర్భావ <signal.h>
#ఒకవేళ defined(__gnu_linux__)
#అన్తర్భావ <syscall.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__APPLE__)
#అన్తర్భావ <unistd.h>
#అన్తర్భావ <mach/mach.h>
#అన్తర్భావ <TargetConditionals.h>
#అయితే_ముగింపు

#ఒకవేళ defined(_WIN32)
#నిర్వచించు WIN32_LEAN_AND_సగటు
#నిర్వచితం_కాకపోతే NOMINMAX
    #నిర్వచించు NOMINMAX
#అయితే_ముగింపు
#అన్తర్భావ <windows.h>
#అయితే_ముగింపు

#నిర్వచించు వాడని GGML_వాడని

uint64_t ggml_లేఖాచిత్రం_తదుపరి_uid(శూన్య) {
#నిర్వచితమైతే _MSC_VER
#ఒకవేళ defined(_WIN32)
    స్థిర అస్థిర LONG counter = 1;
    తిరిగి (uint64_t) InterlockedIncrement(&counter) - 1;
#లేకపోతే
    స్థిర అస్థిర దీర్ఘ దీర్ఘ counter = 1;
    తిరిగి (uint64_t) _InterlockedIncrement64(&counter) - 1;
#అయితే_ముగింపు
#లేకపోతే
    స్థిర uint64_t counter = 1;
    తిరిగి __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
#అయితే_ముగింపు
}

// Needed for ggml_fp32_to_bf16_row()
#ఒకవేళ defined(__AVX512BF16__)
#ఒకవేళ defined(_MSC_VER)
#నిర్వచించు m512i(p) p
#లేకపోతే
#అన్తర్భావ <immintrin.h>
#నిర్వచించు m512i(p) (__m512i)(p)
#అయితే_ముగింపు // defined(_MSC_VER)
#అయితే_ముగింపు // defined(__AVX512BF16__)

#ఒకవేళ defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !లక్ష్యం_OS_TV && !లక్ష్యం_OS_WATCH)

#అన్తర్భావ <unistd.h>
#అన్తర్భావ <sys/types.h>
#అన్తర్భావ <sys/stat.h>
#అన్తర్భావ <sys/wait.h>
#ఒకవేళ defined(__linux__)
#అన్తర్భావ <sys/prctl.h>
#అయితే_ముగింపు

#ఒకవేళ defined(__ANDROID__)
#అన్తర్భావ <unwind.h>
#అన్తర్భావ <dlfcn.h>
#అన్తర్భావ <stdio.h>

నిర్మాణం backtrace_state {
    శూన్య ** current;
    శూన్య ** చివర;
};

స్థిర Unwind_Reason_సంకేతం unwind_callback(నిర్మాణం _Unwind_Context* context, శూన్య* arg) {
    నిర్మాణం backtrace_state * state = (నిర్మాణం backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    ఒకవేళ (pc) {
        ఒకవేళ (state->current == state->చివర) {
            తిరిగి URC_చివర_యొక్క_దొంతర;
        } లేకపోతే {
            *state->current++ = (శూన్య*)pc;
        }
    }
    తిరిగి _URC_NO_REASON;
}

స్థిర శూన్య ggml_print_backtrace_symbols(శూన్య) {
    స్థిరరాశి పూర్ణాంకం max = 100;
    శూన్య* బఫరు[max];

    నిర్మాణం backtrace_state state = {బఫరు, బఫరు + max};
    _Unwind_Backtrace(unwind_callback, &state);

    పూర్ణాంకం లెక్క = state.current - బఫరు;

    కొరకు (పూర్ణాంకం idx = 0; idx < లెక్క; ++idx) {
        స్థిరరాశి శూన్య * addr = బఫరు[idx];
        స్థిరరాశి అక్షరం * చిహ్నం = "";

        Dl_info info;
        ఒకవేళ (dladdr(addr, &info) && info.dli_sname) {
            చిహ్నం = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, చిహ్నం);
    }
}
#లేదావేళ defined(__linux__) && defined(__GLIBC__)
#అన్తర్భావ <execinfo.h>
స్థిర శూన్య ggml_print_backtrace_symbols(శూన్య) {
    శూన్య * trace[100];
    పూర్ణాంకం nptrs = backtrace(trace, పరిమాణం(trace)/పరిమాణం(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#లేదావేళ defined(__APPLE__)
#అన్తర్భావ <execinfo.h>
స్థిర శూన్య ggml_print_backtrace_symbols(శూన్య) {
    శూన్య * trace[100];
    పూర్ణాంకం nptrs = backtrace(trace, పరిమాణం(trace)/పరిమాణం(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#లేకపోతే
స్థిర శూన్య ggml_print_backtrace_symbols(శూన్య) {
    // platform not supported
}
#అయితే_ముగింపు

శూన్య ggml_print_backtrace(శూన్య) {
    స్థిరరాశి అక్షరం * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ఒకవేళ (GGML_NO_BACKTRACE) {
        తిరిగి;
    }
#ఒకవేళ defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    స్థిరరాశి అక్షరం * GGML_BACKTRACE_LLDB = getenv("GGML_BACKTRACE_LLDB");
    ఒకవేళ (!GGML_BACKTRACE_LLDB) {
        fprintf(stderr, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        fprintf(stderr, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        fprintf(stderr, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        తిరిగి;
    }
#అయితే_ముగింపు
#ఒకవేళ defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    size_t కొలత = 0;
    అక్షరం * పంక్తి = ఖాళీ;
    ssize_t నిడివి = 0;
    అయితే ((నిడివి = getline(&పంక్తి, &కొలత, f)) > 0) {
        ఒకవేళ (!అక్షరశ్రేణి_పోలిక(పంక్తి, "TracerPid:", పరిమాణం("TracerPid:") - 1) &&
            (నిడివి != పరిమాణం("TracerPid:\t0\n") - 1 || పంక్తి[నిడివి - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            విడుదల(పంక్తి);
            fclose(f);
            తిరిగి;
        }
    }
    విడుదల(పంక్తి);
    fclose(f);
    పూర్ణాంకం తాళం[2] = { -1, -1 };
    (శూన్య) !pipe(తాళం); // Don't start gdb until after PR_SET_PTRACER
#అయితే_ముగింపు
    స్థిరరాశి పూర్ణాంకం మాతృ_pid = getpid();
    స్థిరరాశి పూర్ణాంకం బిడ్డ_pid = fork();
    ఒకవేళ (బిడ్డ_pid < 0) { // error
#ఒకవేళ defined(__linux__)
        close(తాళం[1]);
        close(తాళం[0]);
#అయితే_ముగింపు
        తిరిగి;
    } లేకపోతే ఒకవేళ (బిడ్డ_pid == 0) { //బిడ్డ
        అక్షరం attach[32];
        snprintf(attach, పరిమాణం(attach), "attach %d", మాతృ_pid);
#ఒకవేళ defined(__linux__)
        close(తాళం[1]);
        (శూన్య) !పఠ(తాళం[0], తాళం, 1);
        close(తాళం[0]);
#అయితే_ముగింపు
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (అక్షరం *) ఖాళీ);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[పరిమాణం("attach ") - 1],
            (అక్షరం *) ఖాళీ);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } లేకపోతే { //మాతృ
#ఒకవేళ defined(__linux__)
        prctl(PR_సమితి_PTRACER, బిడ్డ_pid);
        close(తాళం[1]);
        close(తాళం[0]);
#అయితే_ముగింపు
        waitpid(బిడ్డ_pid, ఖాళీ, 0);
    }
}
#లేకపోతే
శూన్య ggml_print_backtrace(శూన్య) {
    // platform not supported
}
#అయితే_ముగింపు

స్థిర ggml_abort_callback_t g_abort_callback = ఖాళీ;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_సమితి_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    తిరిగి ret_val;
}

శూన్య ggml_abort(స్థిరరాశి అక్షరం * file, పూర్ణాంకం పంక్తి, స్థిరరాశి అక్షరం * fmt, ...) {
    fflush(stdout);

    అక్షరం సందేశం[2048];
    పూర్ణాంకం స్థానభ్రంశం = snprintf(సందేశం, పరిమాణం(సందేశం), "%s:%d: ", file, పంక్తి);

    va_list వాదనలు_2;
    va_start(వాదనలు_2, fmt);
    vsnprintf(సందేశం + స్థానభ్రంశం, పరిమాణం(సందేశం) - స్థానభ్రంశం, fmt, వాదనలు_2);
    va_end(వాదనలు_2);

    ఒకవేళ (g_abort_callback) {
        g_abort_callback(సందేశం);
    } లేకపోతే {
        // default: print error and backtrace to stderr
        fprintf(stderr, "%s\n", సందేశం);
        ggml_print_backtrace();
    }

    abort();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

నిర్మాణం ggml_logger_state {
    ggml_log_callback log_callback;
    శూన్య * log_callback_user_దత్తాంశం;
};
స్థిర నిర్మాణం ggml_logger_state g_logger_state = {ggml_log_callback_default, ఖాళీ};

స్థిర శూన్య ggml_log_అంతర్గత_v(గణన ggml_log_level level, స్థిరరాశి అక్షరం * format, va_list వాదనలు_2) {
    ఒకవేళ (format == ఖాళీ) {
        తిరిగి;
    }
    va_list వాదనలు_copy;
    va_copy(వాదనలు_copy, వాదనలు_2);
    అక్షరం బఫరు[128];
    పూర్ణాంకం len = vsnprintf(బఫరు, 128, format, వాదనలు_2);
    ఒకవేళ (len < 128) {
        g_logger_state.log_callback(level, బఫరు, g_logger_state.log_callback_user_దత్తాంశం);
    } లేకపోతే {
        అక్షరం * buffer2 = (అక్షరం *) సున్నా_జ్ఞాపకం_ఇవ్వు(len + 1, పరిమాణం(అక్షరం));
        vsnprintf(buffer2, len + 1, format, వాదనలు_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_దత్తాంశం);
        విడుదల(buffer2);
    }
    va_end(వాదనలు_copy);
}

శూన్య ggml_log_అంతర్గత(గణన ggml_log_level level, స్థిరరాశి అక్షరం * format, ...) {
    va_list వాదనలు_2;
    va_start(వాదనలు_2, format);
    ggml_log_అంతర్గత_v(level, format, వాదనలు_2);
    va_end(వాదనలు_2);
}

శూన్య ggml_log_callback_default(గణన ggml_log_level level, స్థిరరాశి అక్షరం * పాఠ్యం_2, శూన్య * user_దత్తాంశం) {
    (శూన్య) level;
    (శూన్య) user_దత్తాంశం;
    fputs(పాఠ్యం_2, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#నిర్వచితమైతే GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#అయితే_ముగింపు


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#ఒకవేళ defined(__has_feature)
#ఒకవేళ __has_feature(thread_sanitizer)
#నిర్వచించు GGML_TSAN_ENABLED 1
#అయితే_ముగింపు
#లేకపోతే
#ఒకవేళ defined(__SANITIZE_THREAD__)
#నిర్వచించు GGML_TSAN_ENABLED 1
#అయితే_ముగింపు
#అయితే_ముగింపు

శూన్య * ggml_aligned_malloc(size_t కొలత) {
#ఒకవేళ defined(__s390x__)
    స్థిరరాశి పూర్ణాంకం alignment = 256;
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం alignment = 64;
#అయితే_ముగింపు

#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
    తిరిగి _aligned_malloc(కొలత, alignment);
#లేకపోతే
    ఒకవేళ (కొలత == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        తిరిగి ఖాళీ;
    }
    శూన్య * aligned_జ్ఞాపకం = ఖాళీ;
  #నిర్వచితమైతే GGML_USE_CPU_HBM
    పూర్ణాంకం ఫలితం = hbw_posix_memalign(&aligned_జ్ఞాపకం, alignment, కొలత);
  #లేదావేళ లక్ష్యం_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_వాడని(alignment);
    kern_return_t alloc_స్థితి = vm_allocate((vm_map_t) mach_task_self(), (vm_చిరునామా_t *) &aligned_జ్ఞాపకం, కొలత, VM_FLAGS_ANYWHERE);
    పూర్ణాంకం ఫలితం = EFAULT;
    ఎంపిక (alloc_స్థితి) {
        స్థితి KERN_SUCCESS:
            ఫలితం = 0;
            ఆపు;
        స్థితి KERN_INVALID_చిరునామా:
            ఫలితం = EINVAL;
            ఆపు;
        స్థితి KERN_NO_SPACE:
            ఫలితం = ENOMEM;
            ఆపు;
        అప్రమేయ:
            ఫలితం = EFAULT;
            ఆపు;
    }
  #లేకపోతే
    పూర్ణాంకం ఫలితం = posix_memalign(&aligned_జ్ఞాపకం, alignment, కొలత);
  #అయితే_ముగింపు
    ఒకవేళ (ఫలితం != 0) {
        // Handle allocation failure
        స్థిరరాశి అక్షరం *error_desc = "unknown allocation error";
        ఎంపిక (ఫలితం) {
            స్థితి EINVAL:
                error_desc = "invalid alignment value";
                ఆపు;
            స్థితి ENOMEM:
                error_desc = "insufficient memory";
                ఆపు;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, కొలత/(1024.0*1024.0));
        తిరిగి ఖాళీ;
    }
    తిరిగి aligned_జ్ఞాపకం;
#అయితే_ముగింపు
}

శూన్య ggml_aligned_free(శూన్య * ptr, size_t కొలత) {
    GGML_వాడని(కొలత);
#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#లేదావేళ GGML_USE_CPU_HBM
    ఒకవేళ (ptr != ఖాళీ) {
        hbw_free(ptr);
    }
#లేదావేళ లక్ష్యం_OS_OSX && !defined(GGML_TSAN_ENABLED)
    ఒకవేళ (ptr != ఖాళీ) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_చిరునామా_t)ptr, కొలత);
    }
#లేకపోతే
    విడుదల(ptr);
#అయితే_ముగింపు
}


అంతఃవరుస స్థిర శూన్య * ggml_malloc(size_t కొలత) {
    ఒకవేళ (కొలత == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        తిరిగి ఖాళీ;
    }
    శూన్య * ఫలితం = జ్ఞాపకం_ఇవ్వు(కొలత);
    ఒకవేళ (ఫలితం == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, కొలత/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    తిరిగి ఫలితం;
}

// calloc
అంతఃవరుస స్థిర శూన్య * ggml_calloc(size_t అంకె, size_t కొలత) {
    ఒకవేళ (అంకె == 0 || కొలత == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        తిరిగి ఖాళీ;
    }
    శూన్య * ఫలితం = సున్నా_జ్ఞాపకం_ఇవ్వు(అంకె, కొలత);
    ఒకవేళ (ఫలితం == ఖాళీ) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, కొలత/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    తిరిగి ఫలితం;
}

#నిర్వచించు GGML_MALLOC(కొలత)      ggml_malloc(కొలత)
#నిర్వచించు GGML_CALLOC(అంకె, కొలత) ggml_calloc(అంకె, కొలత)

#నిర్వచించు GGML_FREE(ptr) విడుదల(ptr)

స్థిరరాశి అక్షరం * ggml_స్థితి_గమ్యం_string(గణన ggml_స్థితి స్థితి_2) {
    ఎంపిక (స్థితి_2) {
        స్థితి GGML_స్థితి_ALLOC_FAILED: తిరిగి "GGML status: error (failed to allocate memory)";
        స్థితి GGML_స్థితి_FAILED:       తిరిగి "GGML status: error (operation failed)";
        స్థితి GGML_స్థితి_SUCCESS:      తిరిగి "GGML status: success";
        స్థితి GGML_స్థితి_ABORTED:      తిరిగి "GGML status: warning (operation aborted)";
    }

    తిరిగి "GGML status: unknown";
}

దశాంశం ggml_fp16_గమ్యం_fp32(ggml_fp16_t x) {
#నిర్వచించు ggml_fp16_గమ్యం_fp32 do_not_use_ggml_fp16_గమ్యం_fp32_లోపల_ggml
    తిరిగి GGML_FP16_గమ్యం_FP32(x);
}

ggml_fp16_t ggml_fp32_గమ్యం_fp16(దశాంశం x) {
#నిర్వచించు ggml_fp32_గమ్యం_fp16 do_not_use_ggml_fp32_గమ్యం_fp16_లోపల_ggml
    తిరిగి GGML_FP32_గమ్యం_FP16(x);
}

దశాంశం ggml_bf16_గమ్యం_fp32(ggml_bf16_t x) {
#నిర్వచించు ggml_bf16_గమ్యం_fp32 do_not_use_ggml_bf16_గమ్యం_fp32_లోపల_ggml
    తిరిగి GGML_BF16_గమ్యం_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_గమ్యం_bf16(దశాంశం x) {
#నిర్వచించు ggml_fp32_గమ్యం_bf16 do_not_use_ggml_fp32_గమ్యం_bf16_లోపల_ggml
    తిరిగి GGML_FP32_గమ్యం_BF16(x);
}

శూన్య ggml_fp16_గమ్యం_fp32_అడ్డువరుస(స్థిరరాశి ggml_fp16_t * x, దశాంశం * y, int64_t n) {
    కొరకు (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_గమ్యం_FP32(x[i]);
    }
}

శూన్య ggml_fp32_గమ్యం_fp16_అడ్డువరుస(స్థిరరాశి దశాంశం * x, ggml_fp16_t * y, int64_t n) {
    పూర్ణాంకం i = 0;
    కొరకు (; i < n; ++i) {
        y[i] = GGML_FP32_గమ్యం_FP16(x[i]);
    }
}

శూన్య ggml_bf16_గమ్యం_fp32_అడ్డువరుస(స్థిరరాశి ggml_bf16_t * x, దశాంశం * y, int64_t n) {
    పూర్ణాంకం i = 0;
    కొరకు (; i < n; ++i) {
        y[i] = GGML_BF16_గమ్యం_FP32(x[i]);
    }
}

శూన్య ggml_fp32_గమ్యం_bf16_అడ్డువరుస_ref(స్థిరరాశి దశాంశం * x, ggml_bf16_t * y, int64_t n) {
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_గమ్యం_bf16(x[i]);
    }
}

శూన్య ggml_fp32_గమ్యం_bf16_అడ్డువరుస(స్థిరరాశి దశాంశం * x, ggml_bf16_t * y, int64_t n) {
  పూర్ణాంకం i = 0;
#ఒకవేళ defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  కొరకు (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#అయితే_ముగింపు
    కొరకు (; i < n; i++) {
        y[i] = GGML_FP32_గమ్యం_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    తిరిగి memcmp(guid_a, guid_b, పరిమాణం(ggml_guid)) == 0;
}

స్థిరరాశి అక్షరం * ggml_సంస్కరణ(శూన్య) {
    తిరిగి GGML_సంస్కరణ;
}

స్థిరరాశి అక్షరం * ggml_commit(శూన్య) {
    తిరిగి GGML_COMMIT;
}

//
// timing
//

#ఒకవేళ defined(_MSC_VER) || defined(__MINGW32__)
స్థిర int64_t timer_freq, timer_ప్రారంభం;
స్థిర BOOL CALLBACK ggml_సమయం_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    వాడని(once);
    వాడని(param);
    వాడని(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_ప్రారంభం = t.QuadPart;

    తిరిగి TRUE;
}
శూన్య ggml_సమయం_init(శూన్య) {
    స్థిర INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_సమయం_init_once, ఖాళీ, ఖాళీ);
}
int64_t ggml_సమయం_ms(శూన్య) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    తిరిగి ((t.QuadPart-timer_ప్రారంభం) * 1000) / timer_freq;
}
int64_t ggml_సమయం_us(శూన్య) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    తిరిగి ((t.QuadPart-timer_ప్రారంభం) * 1000000) / timer_freq;
}
#లేకపోతే
శూన్య ggml_సమయం_init(శూన్య) {}
int64_t ggml_సమయం_ms(శూన్య) {
    నిర్మాణం timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    తిరిగి (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_సమయం_us(శూన్య) {
    నిర్మాణం timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    తిరిగి (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#అయితే_ముగింపు

int64_t ggml_చక్రాలు(శూన్య) {
    తిరిగి clock();
}

int64_t ggml_చక్రాలు_చొప్పున_ms(శూన్య) {
    తిరిగి CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#నిర్వచితమైతే _WIN32
స్థిర wchar_t * ggml_mbstowcs(స్థిరరాశి అక్షరం * mbs) {
    పూర్ణాంకం wlen = Multi_Byte_గమ్యం_Wide_Char(CP_UTF8, 0, mbs, -1, ఖాళీ, 0);
    ఒకవేళ (!wlen) {
        errno = EINVAL;
        తిరిగి ఖాళీ;
    }

    wchar_t * wbuf = GGML_MALLOC(wlen * పరిమాణం(wchar_t));
    wlen = Multi_Byte_గమ్యం_Wide_Char(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    ఒకవేళ (!wlen) {
        GGML_FREE(wbuf);
        errno = EINVAL;
        తిరిగి ఖాళీ;
    }

    తిరిగి wbuf;
}
#అయితే_ముగింపు

FILE * ggml_fopen(స్థిరరాశి అక్షరం * fname, స్థిరరాశి అక్షరం * బహుళకం) {
#నిర్వచితమైతే _WIN32
    FILE * file = ఖాళీ;

    // convert fname (UTF-8)
    wchar_t * wfname = ggml_mbstowcs(fname);
    ఒకవేళ (wfname) {
        // convert mode (UTF-8)
        wchar_t * wmode = ggml_mbstowcs(బహుళకం);
        ఒకవేళ (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    తిరిగి file;
#లేకపోతే
    తిరిగి fopen(fname, బహుళకం);
#అయితే_ముగింపు

}

స్థిర స్థిరరాశి నిర్మాణం ggml_type_traits type_traits[GGML_TYPE_లెక్క] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(ద్విగుణ),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(దశాంశం),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(ggml_fp16_t),
        .is_quantized             = false,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) ggml_fp16_గమ్యం_fp32_అడ్డువరుస,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_గమ్యం_fp16_అడ్డువరుస,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_కొలత                = QK1_0,
        .type_కొలత                = పరిమాణం(ఖండం_q1_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_కొలత                = QK2_0,
        .type_కొలత                = పరిమాణం(ఖండం_q2_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_కొలత                = QK4_0,
        .type_కొలత                = పరిమాణం(ఖండం_q4_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_కొలత                = QK4_1,
        .type_కొలత                = పరిమాణం(ఖండం_q4_1),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q4_1,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q4_1_ref,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_కొలత                = QK5_0,
        .type_కొలత                = పరిమాణం(ఖండం_q5_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_కొలత                = QK5_1,
        .type_కొలత                = పరిమాణం(ఖండం_q5_1),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_కొలత                = QK8_0,
        .type_కొలత                = పరిమాణం(ఖండం_q8_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_కొలత                = QK8_1,
        .type_కొలత                = పరిమాణం(ఖండం_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_కొలత                = QK_MXFP4,
        .type_కొలత                = పరిమాణం(ఖండం_mxfp4),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_కొలత                = QK_NVFP4,
        .type_కొలత                = పరిమాణం(ఖండం_nvfp4),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q2_K),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q3_K),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q4_K),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q5_K),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q6_K),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq2_xxs),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq2_xxs,
        .from_float_ref           = ఖాళీ,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq2_xs),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq2_xs,
        .from_float_ref           = ఖాళీ,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq3_xxs),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq3_s),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq2_s),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq1_s),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq1_s,
        .from_float_ref           = ఖాళీ,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq1_m),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq1_m,
        .from_float_ref           = ఖాళీ,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_కొలత                = QK4_NL,
        .type_కొలత                = పరిమాణం(ఖండం_iq4_nl),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_iq4_xs),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_అడ్డువరుస_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_కొలత                = 1,
        .type_కొలత                = పరిమాణం(ggml_bf16_t),
        .is_quantized             = false,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) ggml_bf16_గమ్యం_fp32_అడ్డువరుస,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_గమ్యం_bf16_అడ్డువరుస_ref,
    },
    [31] = { // GGML_TYPE_Q4_0_4_4
        .type_name                = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [32] = { // GGML_TYPE_Q4_0_4_8
        .type_name                = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [33] = { // GGML_TYPE_Q4_0_8_8
        .type_name                = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_tq1_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_కొలత                = QK_K,
        .type_కొలత                = పరిమాణం(ఖండం_tq2_0),
        .is_quantized             = true,
        .గమ్యం_float                 = (ggml_గమ్యం_float_t) dequantize_అడ్డువరుస_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_అడ్డువరుస_tq2_0_ref,
    },
    [36] = { // GGML_TYPE_IQ4_NL_4_4
        .type_name                = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [37] = { // GGML_TYPE_IQ4_NL_4_8
        .type_name                = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
    [38] = { // GGML_TYPE_IQ4_NL_8_8
        .type_name                = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        .blck_కొలత                = 0,
        .type_కొలత                = 0,
        .is_quantized             = false,
    },
};

స్థిరరాశి నిర్మాణం ggml_type_traits * ggml_get_type_traits(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి &type_traits[type];
}

//
// ggml object
//

నిర్మాణం ggml_object {
    size_t offs;
    size_t కొలత;

    నిర్మాణం ggml_object * తదుపరి;

    గణన ggml_object_type type;

    అక్షరం padding[4];
};

స్థిర స్థిరరాశి size_t GGML_OBJECT_కొలత = పరిమాణం(నిర్మాణం ggml_object);

//
// ggml context
//

నిర్మాణం ggml_context {
    size_t mem_కొలత;
    శూన్య * mem_బఫరు;
    bool   mem_బఫరు_owned;
    bool   no_alloc;

    పూర్ణాంకం    n_objects;

    నిర్మాణం ggml_object * objects_begin;
    నిర్మాణం ggml_object * objects_చివర;
};

//
// data types
//

స్థిర స్థిరరాశి అక్షరం * GGML_OP_NAME[GGML_OP_లెక్క] = {
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

static_assert(GGML_OP_లెక్క == 102, "GGML_OP_COUNT != 102");

స్థిర స్థిరరాశి అక్షరం * GGML_OP_చిహ్నం[GGML_OP_లెక్క] = {
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

static_assert(GGML_OP_లెక్క == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_లెక్క == 2, "GGML_OP_POOL_COUNT != 2");

స్థిర స్థిరరాశి అక్షరం * GGML_UNARY_OP_NAME[GGML_UNARY_OP_లెక్క] = {
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

static_assert(GGML_UNARY_OP_లెక్క == 22, "GGML_UNARY_OP_COUNT != 22");

స్థిర స్థిరరాశి అక్షరం * GGML_GLU_OP_NAME[GGML_GLU_OP_లెక్క] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_లెక్క == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(పరిమాణం(నిర్మాణం ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(పరిమాణం(నిర్మాణం ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

శూన్య ggml_print_object(స్థిరరాశి నిర్మాణం ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->కొలత, (స్థిరరాశి శూన్య *) obj->తదుపరి);
}

శూన్య ggml_print_objects(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    నిర్మాణం ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (స్థిరరాశి శూన్య *) ctx);

    అయితే (obj != ఖాళీ) {
        ggml_print_object(obj);
        obj = obj->తదుపరి;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
        ఒకవేళ (tensor->ne[i] <= 0) {
            తిరిగి 0;
        }
    }

    size_t nbytes;
    స్థిరరాశి size_t blck_కొలత = ggml_blck_కొలత(tensor->type);
    ఒకవేళ (blck_కొలత == 1) {
        nbytes = ggml_type_కొలత(tensor->type);
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    లేకపోతే {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_కొలత;
        కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    తిరిగి nbytes;
}

size_t ggml_nbytes_pad(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_కొలత(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి type_traits[type].blck_కొలత;
}

size_t ggml_type_కొలత(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి type_traits[type].type_కొలత;
}

size_t ggml_అడ్డువరుస_కొలత(గణన ggml_type type, int64_t ne) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    ప్రతిజ్ఞా(ne % ggml_blck_కొలత(type) == 0);
    తిరిగి ggml_type_కొలత(type)*ne/ggml_blck_కొలత(type);
}

ద్విగుణ ggml_type_sizef(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి ((ద్విగుణ)(type_traits[type].type_కొలత))/type_traits[type].blck_కొలత;
}

స్థిరరాశి అక్షరం * ggml_type_name(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి type_traits[type].type_name;
}

bool ggml_is_quantized(గణన ggml_type type) {
    ప్రతిజ్ఞా(type >= 0);
    ప్రతిజ్ఞా(type < GGML_TYPE_లెక్క);
    తిరిగి type_traits[type].is_quantized;
}

స్థిరరాశి అక్షరం * ggml_op_name(గణన ggml_op op) {
    తిరిగి GGML_OP_NAME[op];
}

స్థిరరాశి అక్షరం * ggml_op_చిహ్నం(గణన ggml_op op) {
    తిరిగి GGML_OP_చిహ్నం[op];
}

స్థిరరాశి అక్షరం * ggml_unary_op_name(గణన ggml_unary_op op) {
    తిరిగి GGML_UNARY_OP_NAME[op];
}

స్థిరరాశి అక్షరం * ggml_glu_op_name(గణన ggml_glu_op op) {
    తిరిగి GGML_GLU_OP_NAME[op];
}

స్థిరరాశి అక్షరం * ggml_op_desc(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (t->op == GGML_OP_UNARY) {
        గణన ggml_unary_op uop = ggml_get_unary_op(t);
        తిరిగి ggml_unary_op_name(uop);
    }
    ఒకవేళ (t->op == GGML_OP_GLU) {
        గణన ggml_glu_op gop = ggml_get_glu_op(t);
        తిరిగి ggml_glu_op_name(gop);
    }
    తిరిగి ggml_op_name(t->op);
}

size_t ggml_element_కొలత(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_type_కొలత(tensor->type);
}

bool ggml_is_scalar(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_సదిశ(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_మాతృక(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి tensor->ne[3] == 1;
}

పూర్ణాంకం ggml_n_dims(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        ఒకవేళ (tensor->ne[i] > 1) {
            తిరిగి i + 1;
        }
    }
    తిరిగి 1;
}

గణన ggml_type ggml_ftype_గమ్యం_ggml_type(గణన ggml_ftype ftype) {
    గణన ggml_type wtype = GGML_TYPE_లెక్క;

    ఎంపిక (ftype) {
        స్థితి GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    ఆపు;
        స్థితి GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_లెక్క; ఆపు;
        స్థితి GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_లెక్క; ఆపు;
    }

    GGML_ASSERT(wtype != GGML_TYPE_లెక్క);

    తిరిగి wtype;
}

size_t ggml_tensor_overhead(శూన్య) {
    తిరిగి GGML_OBJECT_కొలత + GGML_TENSOR_కొలత;
}

bool ggml_is_transposed(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి tensor->nb[0] > tensor->nb[1];
}

స్థిర bool ggml_is_contiguous_m_n(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం m, పూర్ణాంకం n) {
    size_t తదుపరి_nb = ggml_type_కొలత(tensor->type);
    ఒకవేళ (tensor->ne[0] != ggml_blck_కొలత(tensor->type) && tensor->nb[0] != తదుపరి_nb) {
        తిరిగి false;
    }
    తదుపరి_nb *= tensor->ne[0]/ggml_blck_కొలత(tensor->type);
    కొరకు (పూర్ణాంకం i = 1; i < n; i++) {
        ఒకవేళ (i > m) {
            ఒకవేళ (tensor->ne[i] != 1 && tensor->nb[i] != తదుపరి_nb) {
                తిరిగి false;
            }
            తదుపరి_nb *= tensor->ne[i];
        } లేకపోతే {
            // this dimension does not need to be contiguous
            తదుపరి_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    తిరిగి true;
}

bool ggml_is_contiguous(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_గమ్యం_1(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_గమ్యం_2(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_గమ్యం_3(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_కొలత(tensor->type)/ggml_blck_కొలత(tensor->type);
}

bool ggml_is_permuted(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_కొలత(tensor->type);
}

bool ggml_is_contiguous_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి
        tensor->ne[0] == ggml_blck_కొలత(tensor->type) ||
        tensor->nb[0] == ggml_type_కొలత(tensor->type);
}

స్థిర అంతఃవరుస bool ggml_is_padded_1d(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి
        tensor->nb[0] == ggml_type_కొలత(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_ఖాళీలు(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
        ఒకవేళ (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            తిరిగి true;
        }
    }
    తిరిగి false;
}

bool ggml_are_same_shape(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి ggml_is_ఖాళీలు(t0) ? ggml_is_ఖాళీలు(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

స్థిర అంతఃవరుస bool ggml_can_repeat_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#నిర్వచించు GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

నిర్మాణం ggml_context * ggml_init(నిర్మాణం ggml_init_params params) {
    స్థిర bool is_మొదటి_call = true;

    ggml_critical_section_ప్రారంభం();

    ఒకవేళ (is_మొదటి_call) {
        // initialize time system (required on Windows)
        ggml_సమయం_init();

        is_మొదటి_call = false;
    }

    ggml_critical_section_చివర();

    నిర్మాణం ggml_context * ctx = GGML_MALLOC(పరిమాణం(నిర్మాణం ggml_context));

    // allow to call ggml_init with 0 size
    ఒకవేళ (params.mem_కొలత == 0) {
        params.mem_కొలత = GGML_MEM_ALIGN;
    }

    స్థిరరాశి size_t mem_కొలత = params.mem_బఫరు ? params.mem_కొలత : GGML_PAD(params.mem_కొలత, GGML_MEM_ALIGN);

    *ctx = (నిర్మాణం ggml_context) {
        /*.mem_size           =*/ mem_కొలత,
        /*.mem_buffer         =*/ params.mem_బఫరు ? params.mem_బఫరు : ggml_aligned_malloc(mem_కొలత),
        /*.mem_buffer_owned   =*/ params.mem_బఫరు ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ ఖాళీ,
        /*.objects_end        =*/ ఖాళీ,
    };

    GGML_ASSERT(ctx->mem_బఫరు != ఖాళీ);

    GGML_ASSERT_ALIGNED(ctx->mem_బఫరు);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    తిరిగి ctx;
}

శూన్య ggml_reset(నిర్మాణం ggml_context * ctx) {
    ఒకవేళ (ctx == ఖాళీ) {
        తిరిగి;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = ఖాళీ;
    ctx->objects_చివర   = ఖాళీ;
}

శూన్య ggml_free(నిర్మాణం ggml_context * ctx) {
    ఒకవేళ (ctx == ఖాళీ) {
        తిరిగి;
    }

    ఒకవేళ (ctx->mem_బఫరు_owned) {
        ggml_aligned_free(ctx->mem_బఫరు, ctx->mem_కొలత);
    }

    GGML_FREE(ctx);
}

size_t ggml_వాడిన_mem(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    తిరిగి ctx->objects_చివర == ఖాళీ ? 0 : ctx->objects_చివర->offs + ctx->objects_చివర->కొలత;
}

bool ggml_get_no_alloc(నిర్మాణం ggml_context * ctx) {
    తిరిగి ctx->no_alloc;
}

శూన్య ggml_సమితి_no_alloc(నిర్మాణం ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

శూన్య * ggml_get_mem_బఫరు(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    తిరిగి ctx->mem_బఫరు;
}

size_t ggml_get_mem_కొలత(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    తిరిగి ctx->mem_కొలత;
}

size_t ggml_get_max_tensor_కొలత(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    size_t max_కొలత = 0;

    కొరకు (నిర్మాణం ggml_tensor * tensor = ggml_get_మొదటి_tensor(ctx); tensor != ఖాళీ; tensor = ggml_get_తదుపరి_tensor(ctx, tensor)) {
        size_t బైట్లు = ggml_nbytes(tensor);
        max_కొలత = MAX(max_కొలత, బైట్లు);
    }

    తిరిగి max_కొలత;
}

////////////////////////////////////////////////////////////////////////////////

స్థిర నిర్మాణం ggml_object * ggml_new_object(నిర్మాణం ggml_context * ctx, గణన ggml_object_type type, size_t కొలత) {
    // always insert objects at the end of the context's memory pool
    నిర్మాణం ggml_object * obj_cur = ctx->objects_చివర;

    స్థిరరాశి size_t cur_offs = obj_cur == ఖాళీ ? 0 : obj_cur->offs;
    స్థిరరాశి size_t cur_కొలత = obj_cur == ఖాళీ ? 0 : obj_cur->కొలత;
    స్థిరరాశి size_t cur_చివర  = cur_offs + cur_కొలత;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(కొలత <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    size_t కొలత_needed = GGML_PAD(కొలత, GGML_MEM_ALIGN);

    అక్షరం * స్థిరరాశి mem_బఫరు = ctx->mem_బఫరు;
    నిర్మాణం ggml_object * స్థిరరాశి obj_new = (నిర్మాణం ggml_object *)(mem_బఫరు + cur_చివర);

    // integer overflow checks
    ఒకవేళ (cur_చివర > SIZE_MAX - కొలత_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_చివర, కొలత_needed);
        తిరిగి ఖాళీ;
    }
    ఒకవేళ (cur_చివర + కొలత_needed > SIZE_MAX - GGML_OBJECT_కొలత) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_చివర, కొలత_needed, (size_t) GGML_OBJECT_కొలత);
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (cur_చివర + కొలత_needed + GGML_OBJECT_కొలత > ctx->mem_కొలత) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_చివర + కొలత_needed + GGML_OBJECT_కొలత, ctx->mem_కొలత);
#నిర్వచితం_కాకపోతే NDEBUG
        GGML_ABORT("not enough space in the context's memory pool");
#అయితే_ముగింపు
        తిరిగి ఖాళీ;
    }

    *obj_new = (నిర్మాణం ggml_object) {
        .offs = cur_చివర + GGML_OBJECT_కొలత,
        .కొలత = కొలత_needed,
        .తదుపరి = ఖాళీ,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_బఫరు + obj_new->offs);

    ఒకవేళ (obj_cur != ఖాళీ) {
        obj_cur->తదుపరి = obj_new;
    } లేకపోతే {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_చివర = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    తిరిగి obj_new;
}

స్థిర నిర్మాణం ggml_tensor * ggml_new_tensor_impl(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type      type,
        పూర్ణాంకం                   n_dims,
        స్థిరరాశి int64_t       * ne,
        నిర్మాణం ggml_tensor  * view_src,
        size_t                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_లెక్క);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    ఒకవేళ (view_src != ఖాళీ && view_src->view_src != ఖాళీ) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    కొరకు (పూర్ణాంకం i = 1; i < n_dims; i++) {
        ఒకవేళ (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    size_t దత్తాంశం_కొలత = ggml_అడ్డువరుస_కొలత(type, ne[0]);
    కొరకు (పూర్ణాంకం i = 1; i < n_dims; i++) {
        దత్తాంశం_కొలత *= ne[i];
    }

    GGML_ASSERT(view_src == ఖాళీ || దత్తాంశం_కొలత == 0 || దత్తాంశం_కొలత + view_offs <= ggml_nbytes(view_src));

    శూన్య * దత్తాంశం = view_src != ఖాళీ ? view_src->దత్తాంశం : ఖాళీ;
    ఒకవేళ (దత్తాంశం != ఖాళీ) {
        దత్తాంశం = (అక్షరం *) దత్తాంశం + view_offs;
    }

    size_t obj_alloc_కొలత = 0;

    ఒకవేళ (view_src == ఖాళీ && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_కొలత = దత్తాంశం_కొలత;
    }

    GGML_ASSERT(GGML_TENSOR_కొలత <= SIZE_MAX - obj_alloc_కొలత);

    నిర్మాణం ggml_object * స్థిరరాశి obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_కొలత + obj_alloc_కొలత);
    GGML_ASSERT(obj_new);

    నిర్మాణం ggml_tensor * స్థిరరాశి ఫలితం = (నిర్మాణం ggml_tensor *)((అక్షరం *)ctx->mem_బఫరు + obj_new->offs);

    *ఫలితం = (నిర్మాణం ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ ఖాళీ,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { ఖాళీ },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_కొలత > 0 ? (శూన్య *)(ఫలితం + 1) : దత్తాంశం,
        /*.name         =*/ { 0 },
        /*.extra        =*/ ఖాళీ,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    కొరకు (పూర్ణాంకం i = 0; i < n_dims; i++) {
        ఫలితం->ne[i] = ne[i];
    }

    ఫలితం->nb[0] = ggml_type_కొలత(type);
    ఫలితం->nb[1] = ఫలితం->nb[0]*(ఫలితం->ne[0]/ggml_blck_కొలత(type));
    కొరకు (పూర్ణాంకం i = 2; i < GGML_MAX_DIMS; i++) {
        ఫలితం->nb[i] = ఫలితం->nb[i - 1]*ఫలితం->ne[i - 1];
    }

    ctx->n_objects++;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_new_tensor(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type      type,
        పూర్ణాంకం                   n_dims,
        స్థిరరాశి int64_t       * ne) {
    తిరిగి ggml_new_tensor_impl(ctx, type, n_dims, ne, ఖాళీ, 0);
}

నిర్మాణం ggml_tensor * ggml_new_tensor_1d(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type      type,
        int64_t ne0) {
    తిరిగి ggml_new_tensor(ctx, type, 1, &ne0);
}

నిర్మాణం ggml_tensor * ggml_new_tensor_2d(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    స్థిరరాశి int64_t ne[2] = { ne0, ne1 };
    తిరిగి ggml_new_tensor(ctx, type, 2, ne);
}

నిర్మాణం ggml_tensor * ggml_new_tensor_3d(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    స్థిరరాశి int64_t ne[3] = { ne0, ne1, ne2 };
    తిరిగి ggml_new_tensor(ctx, type, 3, ne);
}

నిర్మాణం ggml_tensor * ggml_new_tensor_4d(
        నిర్మాణం ggml_context * ctx,
        గణన   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    స్థిరరాశి int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    తిరిగి ggml_new_tensor(ctx, type, 4, ne);
}

శూన్య * ggml_new_బఫరు(నిర్మాణం ggml_context * ctx, size_t nbytes) {
    నిర్మాణం ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_బఫరు, nbytes);

    తిరిగి (uint8_t *)ctx->mem_బఫరు + obj->offs;
}

నిర్మాణం ggml_tensor * ggml_dup_tensor(నిర్మాణం ggml_context * ctx, స్థిరరాశి నిర్మాణం ggml_tensor * src) {
    తిరిగి ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

శూన్య ggml_unravel_సూచీ(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    స్థిరరాశి int64_t ne2 = tensor->ne[2];
    స్థిరరాశి int64_t ne1 = tensor->ne[1];
    స్థిరరాశి int64_t ne0 = tensor->ne[0];

    స్థిరరాశి int64_t i3_ = (i/(ne2*ne1*ne0));
    స్థిరరాశి int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    స్థిరరాశి int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    స్థిరరాశి int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    ఒకవేళ (i0) {
        * i0 = i0_;
    }
    ఒకవేళ (i1) {
        * i1 = i1_;
    }
    ఒకవేళ (i2) {
        * i2 = i2_;
    }
    ఒకవేళ (i3) {
        * i3 = i3_;
    }
}

శూన్య * ggml_get_దత్తాంశం(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి tensor->దత్తాంశం;
}

దశాంశం * ggml_get_దత్తాంశం_f32(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    ప్రతిజ్ఞా(tensor->type == GGML_TYPE_F32);
    తిరిగి (దశాంశం *)(tensor->దత్తాంశం);
}

గణన ggml_unary_op ggml_get_unary_op(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    తిరిగి (గణన ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

గణన ggml_glu_op ggml_get_glu_op(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    తిరిగి (గణన ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

స్థిరరాశి అక్షరం * ggml_get_name(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి tensor->name;
}

నిర్మాణం ggml_tensor * ggml_సమితి_name(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి అక్షరం * name) {
    size_t i;
    కొరకు (i = 0; i < పరిమాణం(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    తిరిగి tensor;
}

నిర్మాణం ggml_tensor * ggml_format_name(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి అక్షరం * fmt, ...) {
    va_list వాదనలు_2;
    va_start(వాదనలు_2, fmt);
    vsnprintf(tensor->name, పరిమాణం(tensor->name), fmt, వాదనలు_2);
    va_end(వాదనలు_2);
    తిరిగి tensor;
}

నిర్మాణం ggml_tensor * ggml_view_tensor(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * src) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(ఫలితం, "%s (view)", src->name);

    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        ఫలితం->nb[i] = src->nb[i];
    }

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_get_మొదటి_tensor(స్థిరరాశి నిర్మాణం ggml_context * ctx) {
    నిర్మాణం ggml_object * obj = ctx->objects_begin;

    అక్షరం * స్థిరరాశి mem_బఫరు = ctx->mem_బఫరు;

    అయితే (obj != ఖాళీ) {
        ఒకవేళ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            తిరిగి (నిర్మాణం ggml_tensor *)(mem_బఫరు + obj->offs);
        }

        obj = obj->తదుపరి;
    }

    తిరిగి ఖాళీ;
}

నిర్మాణం ggml_tensor * ggml_get_తదుపరి_tensor(స్థిరరాశి నిర్మాణం ggml_context * ctx, నిర్మాణం ggml_tensor * tensor) {
    నిర్మాణం ggml_object * obj = (నిర్మాణం ggml_object *) ((అక్షరం *)tensor - GGML_OBJECT_కొలత);
    obj = obj->తదుపరి;

    అక్షరం * స్థిరరాశి mem_బఫరు = ctx->mem_బఫరు;

    అయితే (obj != ఖాళీ) {
        ఒకవేళ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            తిరిగి (నిర్మాణం ggml_tensor *)(mem_బఫరు + obj->offs);
        }

        obj = obj->తదుపరి;
    }

    తిరిగి ఖాళీ;
}

నిర్మాణం ggml_tensor * ggml_get_tensor(నిర్మాణం ggml_context * ctx, స్థిరరాశి అక్షరం * name) {
    నిర్మాణం ggml_object * obj = ctx->objects_begin;

    అక్షరం * స్థిరరాశి mem_బఫరు = ctx->mem_బఫరు;

    అయితే (obj != ఖాళీ) {
        ఒకవేళ (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            నిర్మాణం ggml_tensor * cur = (నిర్మాణం ggml_tensor *)(mem_బఫరు + obj->offs);
            ఒకవేళ (strcmp(cur->name, name) == 0) {
                తిరిగి cur;
            }
        }

        obj = obj->తదుపరి;
    }

    తిరిగి ఖాళీ;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

స్థిర నిర్మాణం ggml_tensor * ggml_dup_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_DUP;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_dup(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_dup_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_dup_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_dup_impl(ctx, a, true);
}

// ggml_add

స్థిర నిర్మాణం ggml_tensor * ggml_కూడు_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_కూడు;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_కూడు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_కూడు_impl(ctx, a, b, false);
}

నిర్మాణం ggml_tensor * ggml_కూడు_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_కూడు_impl(ctx, a, b, true);
}

// ggml_add_cast

స్థిర నిర్మాణం ggml_tensor * ggml_కూడు_cast_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        గణన   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_అడ్డువరుసలు(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    ఫలితం->op     = GGML_OP_కూడు;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_కూడు_cast(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        గణన   ggml_type      type) {
    తిరిగి ggml_కూడు_cast_impl(ctx, a, b, type);
}

నిర్మాణం ggml_tensor * ggml_కూడు_id(
            నిర్మాణం ggml_context * ctx,
            నిర్మాణం ggml_tensor  * a,
            నిర్మాణం ggml_tensor  * b,
            నిర్మాణం ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_కూడు_ID;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;
    ఫలితం->src[2] = ids;

    తిరిగి ఫలితం;
}

// ggml_add1

స్థిర నిర్మాణం ggml_tensor * ggml_add1_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_ADD1;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_add1(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_add1_impl(ctx, a, b, false);
}

నిర్మాణం ggml_tensor * ggml_add1_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

స్థిర నిర్మాణం ggml_tensor * ggml_acc_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(స్థానభ్రంశం < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, స్థానభ్రంశం, inplace ? 1 : 0 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_ACC;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_acc(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, స్థానభ్రంశం, false);
}

నిర్మాణం ggml_tensor * ggml_acc_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, స్థానభ్రంశం, true);
}

// ggml_sub

స్థిర నిర్మాణం ggml_tensor * ggml_sub_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SUB;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_sub(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_sub_impl(ctx, a, b, false);
}

నిర్మాణం ggml_tensor * ggml_sub_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

స్థిర నిర్మాణం ggml_tensor * ggml_mul_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_MUL;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_mul(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_mul_impl(ctx, a, b, false);
}

నిర్మాణం ggml_tensor * ggml_mul_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

స్థిర నిర్మాణం ggml_tensor * ggml_div_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_DIV;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_div(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_div_impl(ctx, a, b, false);
}

నిర్మాణం ggml_tensor * ggml_div_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

స్థిర నిర్మాణం ggml_tensor * ggml_sqr_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SQR;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_sqr(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sqr_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_sqr_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

స్థిర నిర్మాణం ggml_tensor * ggml_sqrt_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SQRT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_sqrt(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sqrt_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_sqrt_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

స్థిర నిర్మాణం ggml_tensor * ggml_log_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_LOG;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_log(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_log_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_log_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_log_impl(ctx, a, true);
}

నిర్మాణం ggml_tensor * ggml_expm1(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

నిర్మాణం ggml_tensor * ggml_expm1_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

నిర్మాణం ggml_tensor * ggml_softplus(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

నిర్మాణం ggml_tensor * ggml_softplus_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

స్థిర నిర్మాణం ggml_tensor * ggml_sin_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SIN;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_sin(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sin_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_sin_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_sin_impl(ctx, a, true);
}

// ggml_cos

స్థిర నిర్మాణం ggml_tensor * ggml_cos_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_COS;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_cos(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_cos_impl(ctx, a, false);
}

నిర్మాణం ggml_tensor * ggml_cos_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_cos_impl(ctx, a, true);
}

// ggml_sum

నిర్మాణం ggml_tensor * ggml_కూడిక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, a->type, 1);

    ఫలితం->op     = GGML_OP_కూడిక;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_sum_rows

నిర్మాణం ggml_tensor * ggml_కూడిక_అడ్డువరుసలు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ఫలితం->op     = GGML_OP_కూడిక_అడ్డువరుసలు;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_cumsum

నిర్మాణం ggml_tensor * ggml_cumsum(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_CUMSUM;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_mean

నిర్మాణం ggml_tensor * ggml_సగటు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_సగటు;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_argmax

నిర్మాణం ggml_tensor * ggml_argmax(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_మాతృక(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    ఫలితం->op     = GGML_OP_ARGMAX;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_count_equal

నిర్మాణం ggml_tensor * ggml_లెక్క_equal(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    ఫలితం->op     = GGML_OP_లెక్క_EQUAL;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_repeat

నిర్మాణం ggml_tensor * ggml_repeat(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ఫలితం->op     = GGML_OP_REPEAT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_repeat_4d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    స్థిరరాశి bool can_repeat = ggml_is_ఖాళీలు(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ఫలితం->op     = GGML_OP_REPEAT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_repeat_back

నిర్మాణం ggml_tensor * ggml_repeat_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    ఫలితం->op     = GGML_OP_REPEAT_వెనుక;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_concat

నిర్మాణం ggml_tensor * ggml_concat(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    నిర్మాణం ggml_tensor  * b,
    పూర్ణాంకం                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    కొరకు (పూర్ణాంకం d = 0; d < GGML_MAX_DIMS; ++d) {
        ఒకవేళ (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            కొనసాగు;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0, dim);

    ఫలితం->op     = GGML_OP_CONCAT;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_abs

నిర్మాణం ggml_tensor * ggml_abs(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

నిర్మాణం ggml_tensor * ggml_abs_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

నిర్మాణం ggml_tensor * ggml_sgn(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

నిర్మాణం ggml_tensor * ggml_sgn_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

నిర్మాణం ggml_tensor * ggml_neg(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

నిర్మాణం ggml_tensor * ggml_neg_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

నిర్మాణం ggml_tensor * ggml_అడుగు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_అడుగు);
}

నిర్మాణం ggml_tensor * ggml_అడుగు_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_అడుగు);
}

// ggml_tanh

నిర్మాణం ggml_tensor * ggml_tanh(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

నిర్మాణం ggml_tensor * ggml_tanh_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

నిర్మాణం ggml_tensor * ggml_elu(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

నిర్మాణం ggml_tensor * ggml_elu_inplace(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

నిర్మాణం ggml_tensor * ggml_relu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

నిర్మాణం ggml_tensor * ggml_relu_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

నిర్మాణం ggml_tensor * ggml_leaky_relu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 negative_వాలు,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params(ఫలితం, &negative_వాలు, పరిమాణం(negative_వాలు));

    ఫలితం->op     = GGML_OP_LEAKY_RELU;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_sigmoid

నిర్మాణం ggml_tensor * ggml_sigmoid(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

నిర్మాణం ggml_tensor * ggml_sigmoid_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

నిర్మాణం ggml_tensor * ggml_gelu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

నిర్మాణం ggml_tensor * ggml_gelu_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

నిర్మాణం ggml_tensor * ggml_gelu_erf(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

నిర్మాణం ggml_tensor * ggml_gelu_erf_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

నిర్మాణం ggml_tensor * ggml_gelu_శీఘ్ర(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_GELU_శీఘ్ర);
}

నిర్మాణం ggml_tensor * ggml_gelu_శీఘ్ర_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_శీఘ్ర);
}

// ggml_silu

నిర్మాణం ggml_tensor * ggml_silu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

నిర్మాణం ggml_tensor * ggml_silu_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

నిర్మాణం ggml_tensor * ggml_xielu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం alpha_n,
        దశాంశం alpha_p,
        దశాంశం beta,
        దశాంశం eps) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_సమితి_op_params_f32(ఫలితం, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_సమితి_op_params_f32(ఫలితం, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_సమితి_op_params_f32(ఫలితం, 3, beta);
    ggml_సమితి_op_params_f32(ఫలితం, 4, eps);

    ఫలితం->op     = GGML_OP_UNARY;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_silu_back

నిర్మాణం ggml_tensor * ggml_silu_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SILU_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml hardswish

నిర్మాణం ggml_tensor * ggml_hardswish(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

నిర్మాణం ggml_tensor * ggml_hardsigmoid(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

నిర్మాణం ggml_tensor * ggml_exp(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

నిర్మాణం ggml_tensor * ggml_exp_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

స్థిర నిర్మాణం ggml_tensor * ggml_glu_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        గణన ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    ఒకవేళ (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, ఖాళీ, 0);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) op);
    ggml_సమితి_op_params_i32(ఫలితం, 1, (int32_t) swapped);

    ఫలితం->op     = GGML_OP_GLU;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_floor

నిర్మాణం ggml_tensor * ggml_floor(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

నిర్మాణం ggml_tensor * ggml_floor_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

నిర్మాణం ggml_tensor * ggml_ceil(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

నిర్మాణం ggml_tensor * ggml_ceil_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

నిర్మాణం ggml_tensor * ggml_విడత(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_విడత);
}

నిర్మాణం ggml_tensor * ggml_విడత_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_విడత);
}

//ggml_trunc

నిర్మాణం ggml_tensor * ggml_trunc(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

నిర్మాణం ggml_tensor * ggml_trunc_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

నిర్మాణం ggml_tensor * ggml_glu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_glu_op      op,
        bool                  swapped) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, op, swapped);
}

నిర్మాణం ggml_tensor * ggml_glu_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        గణన ggml_glu_op      op) {
    తిరిగి ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

నిర్మాణం ggml_tensor * ggml_reglu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_REGLU, false);
}

నిర్మాణం ggml_tensor * ggml_reglu_swapped(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_REGLU, true);
}

నిర్మాణం ggml_tensor * ggml_reglu_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

నిర్మాణం ggml_tensor * ggml_geglu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU, false);
}

నిర్మాణం ggml_tensor * ggml_geglu_swapped(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU, true);
}

నిర్మాణం ggml_tensor * ggml_geglu_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

నిర్మాణం ggml_tensor * ggml_swiglu(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_SWIGLU, false);
}

నిర్మాణం ggml_tensor * ggml_swiglu_swapped(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_SWIGLU, true);
}

నిర్మాణం ggml_tensor * ggml_swiglu_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

నిర్మాణం ggml_tensor * ggml_geglu_erf(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU_ERF, false);
}

నిర్మాణం ggml_tensor * ggml_geglu_erf_swapped(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU_ERF, true);
}

నిర్మాణం ggml_tensor * ggml_geglu_erf_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

నిర్మాణం ggml_tensor * ggml_geglu_శీఘ్ర(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU_శీఘ్ర, false);
}

నిర్మాణం ggml_tensor * ggml_geglu_శీఘ్ర_swapped(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_glu_impl(ctx, a, ఖాళీ, GGML_GLU_OP_GEGLU_శీఘ్ర, true);
}

నిర్మాణం ggml_tensor * ggml_geglu_శీఘ్ర_విభజించు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_శీఘ్ర, false);
}

నిర్మాణం ggml_tensor * ggml_swiglu_oai(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 alpha,
        దశాంశం                 హద్దు) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_సమితి_op_params_f32(ఫలితం, 2, alpha);
    ggml_సమితి_op_params_f32(ఫలితం, 3, హద్దు);

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_swiglu_clamp(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 హద్దు) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_సమితి_op_params_f32(ఫలితం, 3, హద్దు);

    తిరిగి ఫలితం;
}

// ggml_norm

స్థిర నిర్మాణం ggml_tensor * ggml_norm_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params(ఫలితం, &eps, పరిమాణం(eps));

    ఫలితం->op     = GGML_OP_NORM;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_norm(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_norm_impl(ctx, a, eps, false);
}

నిర్మాణం ggml_tensor * ggml_norm_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

స్థిర నిర్మాణం ggml_tensor * ggml_rms_norm_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params(ఫలితం, &eps, పరిమాణం(eps));

    ఫలితం->op     = GGML_OP_RMS_NORM;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_rms_norm(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_rms_norm_impl(ctx, a, eps, false);
}

నిర్మాణం ggml_tensor * ggml_rms_norm_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

నిర్మాణం ggml_tensor * ggml_rms_norm_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 eps) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params(ఫలితం, &eps, పరిమాణం(eps));

    ఫలితం->op     = GGML_OP_RMS_NORM_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_group_norm

స్థిర నిర్మాణం ggml_tensor * ggml_సమూహం_norm_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_groups,
        దశాంశం                 eps,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_i32(ఫలితం, 0, n_groups);
    ggml_సమితి_op_params_f32(ఫలితం, 1, eps);

    ఫలితం->op     = GGML_OP_సమూహం_NORM;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_సమూహం_norm(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_groups,
        దశాంశం                 eps) {
    తిరిగి ggml_సమూహం_norm_impl(ctx, a, n_groups, eps, false);
}

నిర్మాణం ggml_tensor * ggml_సమూహం_norm_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_groups,
        దశాంశం                 eps) {
    తిరిగి ggml_సమూహం_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

స్థిర నిర్మాణం ggml_tensor * ggml_l2_norm_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_f32(ఫలితం, 0, eps);

    ఫలితం->op     = GGML_OP_L2_NORM;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_l2_norm(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_l2_norm_impl(ctx, a, eps, false);
}

నిర్మాణం ggml_tensor * ggml_l2_norm_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 eps) {
    తిరిగి ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_సమితి_acc(
        నిర్మాణం ggml_tensor * a,
        గణన ggml_prec       prec) {
    ఎంపిక (a->op) {
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                స్థిరరాశి int32_t prec_i32 = (int32_t) prec;
                ggml_సమితి_op_params_i32(a, 0, prec_i32);
            }
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            {
                స్థిరరాశి int32_t prec_i32 = (int32_t) prec;
                ggml_సమితి_op_params_i32(a, 3, prec_i32);
            }
            ఆపు;
        అప్రమేయ:
            తిరిగి false;
    };

    తిరిగి true;
}

bool ggml_prec_సమితి_src(
        నిర్మాణం ggml_tensor * a,
        గణన ggml_prec       prec,
        పూర్ణాంకం                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    ఎంపిక (a->op) {
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                ఒకవేళ (idx != 1) {
                    తిరిగి false;
                }

                స్థిరరాశి int32_t prec_i32 = (int32_t) prec;
                ggml_సమితి_op_params_i32(a, 2 + idx, prec_i32);
            }
            ఆపు;
        అప్రమేయ:
            తిరిగి false;
    };

    తిరిగి true;
}

// ggml_mul_mat

స్థిర అంతఃవరుస bool ggml_can_mul_mat(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

నిర్మాణం ggml_tensor * ggml_mul_mat(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    స్థిరరాశి int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_MUL_MAT;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

శూన్య ggml_mul_mat_సమితి_prec(
        నిర్మాణం ggml_tensor * a,
        గణన ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    స్థిరరాశి int32_t prec_i32 = (int32_t) prec;

    ggml_సమితి_op_params_i32(a, 0, prec_i32);
}

శూన్య ggml_mul_mat_సమితి_hint(
        నిర్మాణం ggml_tensor * a,
        గణన ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    స్థిరరాశి int32_t hint_i32 = (int32_t) hint;

    ggml_సమితి_op_params_i32(a, 1, hint_i32);
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
నిర్మాణం ggml_tensor * ggml_mul_mat_id(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * as,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    స్థిరరాశి int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_MUL_MAT_ID;
    ఫలితం->src[0] = as;
    ఫలితం->src[1] = b;
    ఫలితం->src[2] = ids;

    తిరిగి ఫలితం;
}

// ggml_out_prod

స్థిర అంతఃవరుస bool ggml_can_బహిర్గతం_prod(స్థిరరాశి నిర్మాణం ggml_tensor * t0, స్థిరరాశి నిర్మాణం ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

నిర్మాణం ggml_tensor * ggml_బహిర్గతం_prod(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_బహిర్గతం_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    స్థిరరాశి int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_బహిర్గతం_PROD;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_scale

స్థిర నిర్మాణం ggml_tensor * ggml_scale_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 s,
        దశాంశం                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    దశాంశం params[2] = { s, b };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_SCALE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_scale(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 s) {
    తిరిగి ggml_scale_impl(ctx, a, s, 0.0, false);
}

నిర్మాణం ggml_tensor * ggml_scale_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 s) {
    తిరిగి ggml_scale_impl(ctx, a, s, 0.0, true);
}

నిర్మాణం ggml_tensor * ggml_scale_bias(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 s,
        దశాంశం                 b) {
    తిరిగి ggml_scale_impl(ctx, a, s, b, false);
}

నిర్మాణం ggml_tensor * ggml_scale_bias_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 s,
        దశాంశం                 b) {
    తిరిగి ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

స్థిర నిర్మాణం ggml_tensor * ggml_సమితి_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(స్థానభ్రంశం < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, స్థానభ్రంశం, inplace ? 1 : 0 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_సమితి;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_సమితి(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, nb1, nb2, nb3, స్థానభ్రంశం, false);
}

నిర్మాణం ggml_tensor * ggml_సమితి_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, nb1, nb2, nb3, స్థానభ్రంశం, true);
}

నిర్మాణం ggml_tensor * ggml_సమితి_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], స్థానభ్రంశం, false);
}

నిర్మాణం ggml_tensor * ggml_సమితి_1d_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], స్థానభ్రంశం, true);
}

నిర్మాణం ggml_tensor * ggml_సమితి_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], స్థానభ్రంశం, false);
}

నిర్మాణం ggml_tensor * ggml_సమితి_2d_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        size_t                nb1,
        size_t                స్థానభ్రంశం) {
    తిరిగి ggml_సమితి_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], స్థానభ్రంశం, true);
}

// ggml_cpy

స్థిర నిర్మాణం ggml_tensor * ggml_cpy_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, b);
    ఒకవేళ (అక్షరశ్రేణి_పొడవు(b->name) > 0) {
        ggml_format_name(ఫలితం, "%s (copy of %s)", b->name, a->name);
    } లేకపోతే {
        ggml_format_name(ఫలితం, "%s (copy)", a->name);
    }

    ఫలితం->op     = GGML_OP_CPY;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_cpy(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor * a,
        నిర్మాణం ggml_tensor * b) {
    తిరిగి ggml_cpy_impl(ctx, a, b);
}

నిర్మాణం ggml_tensor * ggml_cast(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన   ggml_type      type) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(ఫలితం, "%s (copy)", a->name);

    ఫలితం->op     = GGML_OP_CPY;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = ఫలితం; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    తిరిగి ఫలితం;
}

// ggml_cont

స్థిర నిర్మాణం ggml_tensor * ggml_cont_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);
    ggml_format_name(ఫలితం, "%s (cont)", a->name);

    ఫలితం->op     = GGML_OP_CONT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_cont(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor * a) {
    తిరిగి ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API నిర్మాణం ggml_tensor * ggml_cont_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0) {
    తిరిగి ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API నిర్మాణం ggml_tensor * ggml_cont_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    తిరిగి ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API నిర్మాణం ggml_tensor * ggml_cont_3d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    తిరిగి ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

నిర్మాణం ggml_tensor * ggml_cont_4d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(ఫలితం, "%s (cont)", a->name);

    ఫలితం->op     = GGML_OP_CONT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_reshape

నిర్మాణం ggml_tensor * ggml_reshape(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor * a,
        నిర్మాణం ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(ఫలితం, "%s (reshaped)", a->name);

    ఫలితం->op     = GGML_OP_RESHAPE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_reshape_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    స్థిరరాశి int64_t ne[1] = { ne0 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(ఫలితం, "%s (reshaped)", a->name);

    ఫలితం->op     = GGML_OP_RESHAPE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_reshape_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    స్థిరరాశి int64_t ne[2] = { ne0, ne1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(ఫలితం, "%s (reshaped)", a->name);

    ఫలితం->op     = GGML_OP_RESHAPE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_reshape_3d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    స్థిరరాశి int64_t ne[3] = { ne0, ne1, ne2 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(ఫలితం, "%s (reshaped)", a->name);

    ఫలితం->op     = GGML_OP_RESHAPE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_reshape_4d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    స్థిరరాశి int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(ఫలితం, "%s (reshaped)", a->name);

    ఫలితం->op     = GGML_OP_RESHAPE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

స్థిర నిర్మాణం ggml_tensor * ggml_view_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_dims,
        స్థిరరాశి int64_t       * ne,
        size_t                స్థానభ్రంశం) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, స్థానభ్రంశం);
    ggml_format_name(ఫలితం, "%s (view)", a->name);

    ggml_సమితి_op_params(ఫలితం, &స్థానభ్రంశం, పరిమాణం(స్థానభ్రంశం));

    ఫలితం->op     = GGML_OP_VIEW;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_view_1d

నిర్మాణం ggml_tensor * ggml_view_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        size_t                స్థానభ్రంశం) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_impl(ctx, a, 1, &ne0, స్థానభ్రంశం);

    తిరిగి ఫలితం;
}

// ggml_view_2d

నిర్మాణం ggml_tensor * ggml_view_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                స్థానభ్రంశం) {
    స్థిరరాశి int64_t ne[2] = { ne0, ne1 };

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_impl(ctx, a, 2, ne, స్థానభ్రంశం);

    ఫలితం->nb[1] = nb1;
    ఫలితం->nb[2] = ఫలితం->nb[1]*ne1;
    ఫలితం->nb[3] = ఫలితం->nb[2];

    తిరిగి ఫలితం;
}

// ggml_view_3d

నిర్మాణం ggml_tensor * ggml_view_3d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                స్థానభ్రంశం) {
    స్థిరరాశి int64_t ne[3] = { ne0, ne1, ne2 };

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_impl(ctx, a, 3, ne, స్థానభ్రంశం);

    ఫలితం->nb[1] = nb1;
    ఫలితం->nb[2] = nb2;
    ఫలితం->nb[3] = ఫలితం->nb[2]*ne2;

    తిరిగి ఫలితం;
}

// ggml_view_4d

నిర్మాణం ggml_tensor * ggml_view_4d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                స్థానభ్రంశం) {
    స్థిరరాశి int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_impl(ctx, a, 4, ne, స్థానభ్రంశం);

    ఫలితం->nb[1] = nb1;
    ఫలితం->nb[2] = nb2;
    ఫలితం->nb[3] = nb3;

    తిరిగి ఫలితం;
}

// ggml_permute

నిర్మాణం ggml_tensor * ggml_క్రమమార్పు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   axis0,
        పూర్ణాంకం                   axis1,
        పూర్ణాంకం                   axis2,
        పూర్ణాంకం                   axis3) {
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

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);
    ggml_format_name(ఫలితం, "%s (permuted)", a->name);

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

    ఫలితం->ne[0] = ne[0];
    ఫలితం->ne[1] = ne[1];
    ఫలితం->ne[2] = ne[2];
    ఫలితం->ne[3] = ne[3];

    ఫలితం->nb[0] = nb[0];
    ఫలితం->nb[1] = nb[1];
    ఫలితం->nb[2] = nb[2];
    ఫలితం->nb[3] = nb[3];

    ఫలితం->op     = GGML_OP_క్రమమార్పు;
    ఫలితం->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    తిరిగి ఫలితం;
}

// ggml_transpose

నిర్మాణం ggml_tensor * ggml_transpose(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);
    ggml_format_name(ఫలితం, "%s (transposed)", a->name);

    ఫలితం->ne[0] = a->ne[1];
    ఫలితం->ne[1] = a->ne[0];

    ఫలితం->nb[0] = a->nb[1];
    ఫలితం->nb[1] = a->nb[0];

    ఫలితం->op     = GGML_OP_TRANSPOSE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_get_rows

నిర్మాణం ggml_tensor * ggml_get_అడ్డువరుసలు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    గణన ggml_type type = GGML_TYPE_F32;
    ఒకవేళ (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    ఫలితం->op     = GGML_OP_GET_అడ్డువరుసలు;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_get_rows_back

నిర్మాణం ggml_tensor * ggml_get_అడ్డువరుసలు_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_మాతృక(a) && ggml_is_సదిశ(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_మాతృక(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    ఫలితం->op     = GGML_OP_GET_అడ్డువరుసలు_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_set_rows

నిర్మాణం ggml_tensor * ggml_సమితి_అడ్డువరుసలు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[2] == b->ne[2]);
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    GGML_ASSERT(b->ne[1] == c->ne[0]);
    GGML_ASSERT(b->ne[2] % c->ne[1] == 0);
    GGML_ASSERT(b->ne[3] % c->ne[2] == 0);
    GGML_ASSERT(c->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I64 || c->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(a));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(b));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_సమితి_అడ్డువరుసలు;
    ఫలితం->src[0] = b;
    ఫలితం->src[1] = c;
    ఫలితం->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    తిరిగి ఫలితం;
}

// ggml_diag

నిర్మాణం ggml_tensor * ggml_diag(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    స్థిరరాశి int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, 4, ne);

    ఫలితం->op     = GGML_OP_DIAG;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_diag_mask_inf

స్థిర నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_inf_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_DIAG_ముసుగు_INF;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_inf(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past) {
    తిరిగి ggml_diag_ముసుగు_inf_impl(ctx, a, n_past, false);
}

నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_inf_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past) {
    తిరిగి ggml_diag_ముసుగు_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

స్థిర నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_zero_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_DIAG_ముసుగు_ZERO;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_zero(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past) {
    తిరిగి ggml_diag_ముసుగు_zero_impl(ctx, a, n_past, false);
}

నిర్మాణం ggml_tensor * ggml_diag_ముసుగు_zero_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_past) {
    తిరిగి ggml_diag_ముసుగు_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

స్థిర నిర్మాణం ggml_tensor * ggml_clamp_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        దశాంశం                 min,
        దశాంశం                 max,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    దశాంశం params[] = { min, max };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_CLAMP;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_clamp(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    దశాంశం                 min,
    దశాంశం                 max) {
    తిరిగి ggml_clamp_impl(ctx, a, min, max, false);
}

నిర్మాణం ggml_tensor * ggml_clamp_inplace(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    దశాంశం                 min,
    దశాంశం                 max) {
    తిరిగి ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

స్థిర నిర్మాణం ggml_tensor * ggml_soft_max_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * ముసుగు,
        దశాంశం                 scale,
        దశాంశం                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    ఒకవేళ (ముసుగు) {
        GGML_ASSERT(ముసుగు->type == GGML_TYPE_F16 || ముసుగు->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(ముసుగు));
        GGML_ASSERT(ముసుగు->ne[0] == a->ne[0]);
        GGML_ASSERT(ముసుగు->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%ముసుగు->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%ముసుగు->ne[3] == 0);
    }

    ఒకవేళ (max_bias > 0.0f) {
        GGML_ASSERT(ముసుగు);
    }

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    దశాంశం params[] = { scale, max_bias };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_SOFT_MAX;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = ముసుగు;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_soft_max(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_soft_max_impl(ctx, a, ఖాళీ, 1.0f, 0.0f, false);
}

నిర్మాణం ggml_tensor * ggml_soft_max_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a) {
    తిరిగి ggml_soft_max_impl(ctx, a, ఖాళీ, 1.0f, 0.0f, true);
}

నిర్మాణం ggml_tensor * ggml_soft_max_ext(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * ముసుగు,
        దశాంశం                 scale,
        దశాంశం                 max_bias) {
    తిరిగి ggml_soft_max_impl(ctx, a, ముసుగు, scale, max_bias, false);
}

నిర్మాణం ggml_tensor * ggml_soft_max_ext_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * ముసుగు,
        దశాంశం                 scale,
        దశాంశం                 max_bias) {
    తిరిగి ggml_soft_max_impl(ctx, a, ముసుగు, scale, max_bias, true);
}

శూన్య ggml_soft_max_కూడు_sinks(
        నిర్మాణం ggml_tensor * a,
        నిర్మాణం ggml_tensor * sinks) {
    ఒకవేళ (!sinks) {
        a->src[2] = ఖాళీ;
        తిరిగి;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == ఖాళీ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

స్థిర నిర్మాణం ggml_tensor * ggml_soft_max_ext_వెనుక_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 scale,
        దశాంశం                 max_bias,
        bool                  inplace) {
    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_SOFT_MAX_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    జ్ఞాపకం_నకలు((దశాంశం *) ఫలితం->op_params + 0, &scale,    పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు((దశాంశం *) ఫలితం->op_params + 1, &max_bias, పరిమాణం(దశాంశం));

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_soft_max_ext_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 scale,
        దశాంశం                 max_bias) {
    తిరిగి ggml_soft_max_ext_వెనుక_impl(ctx, a, b, scale, max_bias, false);
}

నిర్మాణం ggml_tensor * ggml_soft_max_ext_వెనుక_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        దశాంశం                 scale,
        దశాంశం                 max_bias) {
    తిరిగి ggml_soft_max_ext_వెనుక_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

స్థిర నిర్మాణం ggml_tensor * ggml_rope_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   sections[GGML_MROPE_SECTIONS],
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((బహుళకం & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_సదిశ(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_వాడిన = బహుళకం & GGML_ROPE_TYPE_MROPE;
    ఒకవేళ (mrope_వాడిన) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } లేకపోతే {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    ఒకవేళ (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, బహుళకం, /*n_ctx*/ 0, n_ctx_orig };
    జ్ఞాపకం_నకలు(params +  5, &freq_పునాది,    పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(params +  6, &freq_scale,   పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(params +  7, &ext_గుణకం,   పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(params +  8, &attn_గుణకం,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(params +  9, &beta_fast,    పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(params + 10, &beta_slow,    పరిమాణం(దశాంశం));
    ఒకవేళ (mrope_వాడిన && sections) {
        జ్ఞాపకం_నకలు(params + 11, sections,  పరిమాణం(int32_t) * GGML_MROPE_SECTIONS);
    } లేకపోతే {
        memset(params + 11, 0,         పరిమాణం(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_ROPE;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;
    ఫలితం->src[2] = c;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_rope(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, ఖాళీ, n_dims, ఖాళీ, బహుళకం, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

నిర్మాణం ggml_tensor * ggml_rope_multi(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   sections[GGML_MROPE_SECTIONS],
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, false
    );
}

నిర్మాణం ggml_tensor * ggml_rope_multi_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   sections[GGML_MROPE_SECTIONS],
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, true
    );
}

నిర్మాణం ggml_tensor * ggml_rope_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, ఖాళీ, n_dims, ఖాళీ, బహుళకం, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

నిర్మాణం ggml_tensor * ggml_rope_ext(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, c, n_dims, ఖాళీ, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, false
    );
}

నిర్మాణం ggml_tensor * ggml_rope_ext_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, c, n_dims, ఖాళీ, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, true
    );
}

నిర్మాణం ggml_tensor * ggml_rope_custom(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, ఖాళీ, n_dims, ఖాళీ, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, false
    );
}

నిర్మాణం ggml_tensor * ggml_rope_custom_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    తిరిగి ggml_rope_impl(
        ctx, a, b, ఖాళీ, n_dims, ఖాళీ, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale,
        ext_గుణకం, attn_గుణకం, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
స్థిర దశాంశం ggml_rope_yarn_corr_dim(పూర్ణాంకం n_dims, పూర్ణాంకం n_ctx_orig, దశాంశం n_rot, దశాంశం పునాది) {
    తిరిగి n_dims * logf(n_ctx_orig / (n_rot * 2 * (దశాంశం)M_PI)) / (2 * logf(పునాది));
}

శూన్య ggml_rope_yarn_corr_dims(
    పూర్ణాంకం n_dims, పూర్ణాంకం n_ctx_orig, దశాంశం freq_పునాది, దశాంశం beta_fast, దశాంశం beta_slow, దశాంశం dims[2]
) {
    // start and end correction dims
    దశాంశం ప్రారంభం = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_పునాది));
    దశాంశం చివర   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_పునాది));
    dims[0] = MAX(0, ప్రారంభం);
    dims[1] = MIN(n_dims - 1, చివర);
}

// ggml_rope_back

నిర్మాణం ggml_tensor * ggml_rope_ext_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_rope_ext(
        ctx, a, b, c, n_dims, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
    ఫలితం->op = GGML_OP_ROPE_వెనుక;
    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_rope_multi_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c,
        పూర్ణాంకం                   n_dims,
        పూర్ణాంకం                   sections[4],
        పూర్ణాంకం                   బహుళకం,
        పూర్ణాంకం                   n_ctx_orig,
        దశాంశం                 freq_పునాది,
        దశాంశం                 freq_scale,
        దశాంశం                 ext_గుణకం,
        దశాంశం                 attn_గుణకం,
        దశాంశం                 beta_fast,
        దశాంశం                 beta_slow) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, బహుళకం, n_ctx_orig, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
    ఫలితం->op = GGML_OP_ROPE_వెనుక;
    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_rope_సమితి_స్థానభ్రంశం(
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_వెనుక);
    GGML_ASSERT(n_offs >= 0);

    స్థిరరాశి int32_t బహుళకం = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(బహుళకం != GGML_ROPE_TYPE_VISION);

    ggml_సమితి_op_params_i32(a, 15, n_offs);
    తిరిగి a;
}

స్థిర int64_t ggml_calc_conv_output_కొలత(int64_t ins, int64_t ks, పూర్ణాంకం s, పూర్ణాంకం p, పూర్ణాంకం d) {
    తిరిగి (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
నిర్మాణం ggml_tensor * ggml_im2col(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   d0,
        పూర్ణాంకం                   d1,
        bool                  is_2D,
        గణన ggml_type        dst_type) {
    ఒకవేళ (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } లేకపోతే {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    స్థిరరాశి int64_t OH = is_2D ? ggml_calc_conv_output_కొలత(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    స్థిరరాశి int64_t OW =         ggml_calc_conv_output_కొలత(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    స్థిరరాశి int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_IM2_నిలువు;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_im2col_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        int64_t             * ne,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   d0,
        పూర్ణాంకం                   d1,
        bool                  is_2D) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_IM2_నిలువు_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_conv_1d

నిర్మాణం ggml_tensor * ggml_conv_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   d0) {
    నిర్మాణం ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    నిర్మాణం ggml_tensor * ఫలితం =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    ఫలితం = ggml_reshape_3d(ctx, ఫలితం, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    ఫలితం = ggml_cont(ctx, ggml_క్రమమార్పు(ctx, ఫలితం, 0, 2, 1, 3));                // [N, OC, OL]

    తిరిగి ఫలితం;
}

// ggml_conv_1d_ph

నిర్మాణం ggml_tensor* ggml_conv_1d_ph(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s,
        పూర్ణాంకం                   d) {
    తిరిగి ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

నిర్మాణం ggml_tensor * ggml_conv_1d_dw(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   d0) {
    నిర్మాణం ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    నిర్మాణం ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_mul_mat(ctx, im2col, a);

    ఫలితం = ggml_reshape_3d(ctx, ఫలితం, ఫలితం->ne[0], ఫలితం->ne[2], 1);

    తిరిగి ఫలితం;
}

// ggml_conv_1d_dw_ph

నిర్మాణం ggml_tensor * ggml_conv_1d_dw_ph(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   d0) {
    తిరిగి ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

నిర్మాణం ggml_tensor * ggml_col2im_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   oc,
        పూర్ణాంకం                   p0) {
    GGML_ASSERT(ggml_is_మాతృక(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    స్థిరరాశి int64_t K_OC = a->ne[0];
    స్థిరరాశి int64_t T_లోపల = a->ne[1];
    స్థిరరాశి int64_t K = K_OC / oc;
    స్థిరరాశి int64_t T_బహిర్గతం = (T_లోపల - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_బహిర్గతం > 0);

    స్థిరరాశి int64_t ne[4] = { T_బహిర్గతం, oc, 1, 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_COL2IM_1D;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_conv_transpose_1d

స్థిర int64_t ggml_calc_conv_transpose_1d_output_కొలత(int64_t ins, int64_t ks, పూర్ణాంకం s, పూర్ణాంకం p, పూర్ణాంకం d) {
    తిరిగి (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API నిర్మాణం ggml_tensor * ggml_conv_transpose_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   d0) {
    GGML_ASSERT(ggml_is_మాతృక(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    స్థిరరాశి int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_కొలత(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_CONV_TRANSPOSE_1D;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
నిర్మాణం ggml_tensor * ggml_conv_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   d0,
        పూర్ణాంకం                   d1) {
    నిర్మాణం ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    నిర్మాణం ggml_tensor * ఫలితం =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    ఫలితం = ggml_reshape_4d(ctx, ఫలితం, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    ఫలితం = ggml_cont(ctx, ggml_క్రమమార్పు(ctx, ఫలితం, 0, 1, 3, 2)); // [N, OC, OH, OW]


    తిరిగి ఫలితం;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
నిర్మాణం ggml_tensor * ggml_im2col_3d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        int64_t               IC,
        పూర్ణాంకం                   s0, // stride width
        పూర్ణాంకం                   s1, // stride height
        పూర్ణాంకం                   s2, // stride depth
        పూర్ణాంకం                   p0, // padding width
        పూర్ణాంకం                   p1, // padding height
        పూర్ణాంకం                   p2, // padding depth
        పూర్ణాంకం                   d0, // dilation width
        పూర్ణాంకం                   d1, // dilation height
        పూర్ణాంకం                   d2, // dilation depth
        గణన ggml_type        dst_type) {
    స్థిరరాశి int64_t N = b->ne[3] / IC;
    స్థిరరాశి int64_t ID = b->ne[2];
    స్థిరరాశి int64_t IH = b->ne[1];
    స్థిరరాశి int64_t IW = b->ne[0];

    స్థిరరాశి int64_t OC = a->ne[3] / IC;
    వాడని(OC);
    స్థిరరాశి int64_t KD = a->ne[2];
    స్థిరరాశి int64_t KH = a->ne[1];
    స్థిరరాశి int64_t KW = a->ne[0];
    స్థిరరాశి int64_t OD = ggml_calc_conv_output_కొలత(ID, KD, s2, p2, d2);
    స్థిరరాశి int64_t OH = ggml_calc_conv_output_కొలత(IH, KH, s1, p1, d1);
    స్థిరరాశి int64_t OW = ggml_calc_conv_output_కొలత(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    స్థిరరాశి int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_IM2_నిలువు_3_D;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
నిర్మాణం ggml_tensor * ggml_conv_3d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        int64_t               IC,
        పూర్ణాంకం                   s0, // stride width
        పూర్ణాంకం                   s1, // stride height
        పూర్ణాంకం                   s2, // stride depth
        పూర్ణాంకం                   p0, // padding width
        పూర్ణాంకం                   p1, // padding height
        పూర్ణాంకం                   p2, // padding depth
        పూర్ణాంకం                   d0, // dilation width
        పూర్ణాంకం                   d1, // dilation height
        పూర్ణాంకం                   d2  // dilation depth
        ) {
    నిర్మాణం ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    నిర్మాణం ggml_tensor * ఫలితం =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    ఫలితం = ggml_reshape_4d(ctx, ఫలితం, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    ఫలితం = ggml_cont(ctx, ggml_క్రమమార్పు(ctx, ఫలితం, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    ఫలితం = ggml_reshape_4d(ctx, ఫలితం, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    తిరిగి ఫలితం;
}

// ggml_conv_2d_sk_p0

నిర్మాణం ggml_tensor * ggml_conv_2d_sk_p0(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

నిర్మాణం ggml_tensor * ggml_conv_2d_s1_ph(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    తిరిగి ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

నిర్మాణం ggml_tensor * ggml_conv_2d_dw(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   d0,
        పూర్ణాంకం                   d1) {
    నిర్మాణం ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    నిర్మాణం ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    నిర్మాణం ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    నిర్మాణం ggml_tensor * ఫలితం = ggml_mul_mat(ctx, new_a, new_b);
    ఫలితం = ggml_reshape_4d(ctx, ఫలితం, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    తిరిగి ఫలితం;
}

// ggml_conv_2d_dw_direct

నిర్మాణం ggml_tensor * ggml_conv_2d_dw_ప్రత్యక్ష(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   stride0,
        పూర్ణాంకం                   stride1,
        పూర్ణాంకం                   pad0,
        పూర్ణాంకం                   pad1,
        పూర్ణాంకం                   dilation0,
        పూర్ణాంకం                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_కొలత(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_కొలత(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, b->type, 4, ne);

    ఒకవేళ (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        స్థిరరాశి int64_t type_కొలత = ggml_type_కొలత(ఫలితం->type);
        GGML_ASSERT(ggml_blck_కొలత(ఫలితం->type) == 1);
        ఫలితం->nb[0] = ఫలితం->ne[2] * type_కొలత;
        ఫలితం->nb[1] = ఫలితం->ne[0] * ఫలితం->nb[0];
        ఫలితం->nb[2] = type_కొలత;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_CONV_2D_DW;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;
    తిరిగి ఫలితం;
}

// ggml_conv_2d_direct

నిర్మాణం ggml_tensor * ggml_conv_2d_ప్రత్యక్ష(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        నిర్మాణం ggml_tensor  * b,   // input data [W, H, C, N]
        పూర్ణాంకం                   s0,  // stride dimension 0
        పూర్ణాంకం                   s1,  // stride dimension 1
        పూర్ణాంకం                   p0,  // padding dimension 0
        పూర్ణాంకం                   p1,  // padding dimension 1
        పూర్ణాంకం                   d0,  // dilation dimension 0
        పూర్ణాంకం                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_కొలత(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_కొలత(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0, s0);
    ggml_సమితి_op_params_i32(ఫలితం, 1, s1);
    ggml_సమితి_op_params_i32(ఫలితం, 2, p0);
    ggml_సమితి_op_params_i32(ఫలితం, 3, p1);
    ggml_సమితి_op_params_i32(ఫలితం, 4, d0);
    ggml_సమితి_op_params_i32(ఫలితం, 5, d1);

    ఫలితం->op = GGML_OP_CONV_2D;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_conv_3d_direct

నిర్మాణం ggml_tensor * ggml_conv_3d_ప్రత్యక్ష(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        పూర్ణాంకం                   s2,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   p2,
        పూర్ణాంకం                   d0,
        పూర్ణాంకం                   d1,
        పూర్ణాంకం                   d2,
        పూర్ణాంకం                   c,
        పూర్ణాంకం                   n,
        పూర్ణాంకం                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_కొలత(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_కొలత(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_కొలత(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0,  s0);
    ggml_సమితి_op_params_i32(ఫలితం, 1,  s1);
    ggml_సమితి_op_params_i32(ఫలితం, 2,  s2);
    ggml_సమితి_op_params_i32(ఫలితం, 3,  p0);
    ggml_సమితి_op_params_i32(ఫలితం, 4,  p1);
    ggml_సమితి_op_params_i32(ఫలితం, 5,  p2);
    ggml_సమితి_op_params_i32(ఫలితం, 6,  d0);
    ggml_సమితి_op_params_i32(ఫలితం, 7,  d1);
    ggml_సమితి_op_params_i32(ఫలితం, 8,  d2);
    ggml_సమితి_op_params_i32(ఫలితం, 9,  c);
    ggml_సమితి_op_params_i32(ఫలితం, 10, n);
    ggml_సమితి_op_params_i32(ఫలితం, 11, oc);

    ఫలితం->op = GGML_OP_CONV_3D;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_conv_transpose_2d_p0

స్థిర int64_t ggml_calc_conv_transpose_output_కొలత(int64_t ins, int64_t ks, పూర్ణాంకం s, పూర్ణాంకం p) {
    తిరిగి (ins - 1) * s - 2 * p + ks;
}

నిర్మాణం ggml_tensor * ggml_conv_transpose_2d_p0(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        పూర్ణాంకం                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    స్థిరరాశి int64_t ne[4] = {
        ggml_calc_conv_transpose_output_కొలత(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_కొలత(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    నిర్మాణం ggml_tensor* ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0, stride);

    ఫలితం->op     = GGML_OP_CONV_TRANSPOSE_2D;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_pool_*

స్థిర int64_t ggml_calc_pool_output_కొలత(int64_t ins, పూర్ణాంకం ks, పూర్ణాంకం s, దశాంశం p) {
    తిరిగి (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

నిర్మాణం ggml_tensor * ggml_pool_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_op_pool     op,
        పూర్ణాంకం                   k0,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   p0) {
    స్థిరరాశి int64_t ne[4] = {
        ggml_calc_pool_output_కొలత(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_POOL_1D;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_pool_2d

నిర్మాణం ggml_tensor * ggml_pool_2d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_op_pool     op,
        పూర్ణాంకం                   k0,
        పూర్ణాంకం                   k1,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        దశాంశం                 p0,
        దశాంశం                 p1) {
    నిర్మాణం ggml_tensor * ఫలితం;
    స్థిరరాశి int64_t ne[4] = {
        ggml_calc_pool_output_కొలత(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_కొలత(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_POOL_2D;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_pool_2d_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * af,
        గణన ggml_op_pool     op,
        పూర్ణాంకం                   k0,
        పూర్ణాంకం                   k1,
        పూర్ణాంకం                   s0,
        పూర్ణాంకం                   s1,
        దశాంశం                 p0,
        దశాంశం                 p1) {
    నిర్మాణం ggml_tensor * ఫలితం;
    ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_POOL_2_D_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = af;

    తిరిగి ఫలితం;
}

// ggml_upscale / ggml_interpolate

స్థిర నిర్మాణం ggml_tensor * ggml_interpolate_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              బహుళకం) {
    GGML_ASSERT((బహుళకం & 0xFF) < GGML_SCALE_బహుళకం_లెక్క);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(బహుళకం & GGML_SCALE_FLAG_ANTIALIAS) || (బహుళకం & 0xFF) == GGML_SCALE_బహుళకం_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t)బహుళకం);

    ఫలితం->op     = GGML_OP_UPSCALE;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_upscale(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   scale_గుణకం,
        గణన ggml_scale_బహుళకం  బహుళకం) {
    GGML_ASSERT(scale_గుణకం > 1);
    తిరిగి ggml_interpolate_impl(ctx, a, a->ne[0] * scale_గుణకం, a->ne[1] * scale_గుణకం, a->ne[2], a->ne[3], బహుళకం);
}

నిర్మాణం ggml_tensor * ggml_upscale_ext(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   ne0,
        పూర్ణాంకం                   ne1,
        పూర్ణాంకం                   ne2,
        పూర్ణాంకం                   ne3,
        గణన ggml_scale_బహుళకం  బహుళకం) {
    తిరిగి ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, బహుళకం);
}

నిర్మాణం ggml_tensor * ggml_interpolate(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              బహుళకం) {
    తిరిగి ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, బహుళకం);
}

// ggml_pad

నిర్మాణం ggml_tensor * ggml_pad(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   p2,
        పూర్ణాంకం                   p3) {
    తిరిగి ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

నిర్మాణం ggml_tensor * ggml_pad_circular(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1,
        పూర్ణాంకం                   p2,
        పూర్ణాంకం                   p3) {
    తిరిగి ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

నిర్మాణం ggml_tensor * ggml_pad_ext(
            నిర్మాణం ggml_context * ctx,
            నిర్మాణం ggml_tensor  * a,
            పూర్ణాంకం                  lp0,
            పూర్ణాంకం                  rp0,
            పూర్ణాంకం                  lp1,
            పూర్ణాంకం                  rp1,
            పూర్ణాంకం                  lp2,
            పూర్ణాంకం                  rp2,
            పూర్ణాంకం                  lp3,
            పూర్ణాంకం                  rp3
            ) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_సమితి_op_params_i32(ఫలితం, 0, lp0);
    ggml_సమితి_op_params_i32(ఫలితం, 1, rp0);
    ggml_సమితి_op_params_i32(ఫలితం, 2, lp1);
    ggml_సమితి_op_params_i32(ఫలితం, 3, rp1);
    ggml_సమితి_op_params_i32(ఫలితం, 4, lp2);
    ggml_సమితి_op_params_i32(ఫలితం, 5, rp2);
    ggml_సమితి_op_params_i32(ఫలితం, 6, lp3);
    ggml_సమితి_op_params_i32(ఫలితం, 7, rp3);
    ggml_సమితి_op_params_i32(ఫలితం, 8, 0); // not circular by default


    ఫలితం->op     = GGML_OP_PAD;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_pad_ext_circular

నిర్మాణం ggml_tensor * ggml_pad_ext_circular(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                  lp0,
        పూర్ణాంకం                  rp0,
        పూర్ణాంకం                  lp1,
        పూర్ణాంకం                  rp1,
        పూర్ణాంకం                  lp2,
        పూర్ణాంకం                  rp2,
        పూర్ణాంకం                  lp3,
        పూర్ణాంకం                  rp3
        ) {
    నిర్మాణం ggml_tensor * ఫలితం = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_సమితి_op_params_i32(ఫలితం, 8, 1); // circular
    తిరిగి ఫలితం;
}

// ggml_pad_reflect_1d

నిర్మాణం ggml_tensor * ggml_pad_reflect_1d(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   p0,
        పూర్ణాంకం                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_PAD_REFLECT_1D;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_roll

నిర్మాణం ggml_tensor * ggml_roll(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   shift0,
        పూర్ణాంకం                   shift1,
        పూర్ణాంకం                   shift2,
        పూర్ణాంకం                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_కొలత(a->type));
    GGML_ASSERT(నిరపేక్ష(shift0) < a->ne[0]);
    GGML_ASSERT(నిరపేక్ష(shift1) < a->ne[1]);
    GGML_ASSERT(నిరపేక్ష(shift2) < a->ne[2]);
    GGML_ASSERT(నిరపేక్ష(shift3) < a->ne[3]);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_i32(ఫలితం, 0, shift0);
    ggml_సమితి_op_params_i32(ఫలితం, 1, shift1);
    ggml_సమితి_op_params_i32(ఫలితం, 2, shift2);
    ggml_సమితి_op_params_i32(ఫలితం, 3, shift3);

    ఫలితం->op     = GGML_OP_ROLL;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_timestep_embedding

నిర్మాణం ggml_tensor * ggml_timestep_embedding(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * timesteps,
        పూర్ణాంకం                   dim,
        పూర్ణాంకం                   max_period) {

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_సమితి_op_params_i32(ఫలితం, 0, dim);
    ggml_సమితి_op_params_i32(ఫలితం, 1, max_period);

    ఫలితం->op     = GGML_OP_TIMESTEP_EMBEDDING;
    ఫలితం->src[0] = timesteps;

    తిరిగి ఫలితం;
}

// ggml_tri

నిర్మాణం ggml_tensor * ggml_tri(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    గణన ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_i32(ఫలితం, 0, type);

    ఫలితం->op = GGML_OP_TRI;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_fill

స్థిర నిర్మాణం ggml_tensor * ggml_fill_impl(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    దశాంశం                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_f32(ఫలితం, 0, c);

    ఫలితం->op = GGML_OP_FILL;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_fill(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    దశాంశం                 c) {
    తిరిగి ggml_fill_impl(ctx, a, c, false);
}

నిర్మాణం ggml_tensor * ggml_fill_inplace(
    నిర్మాణం ggml_context * ctx,
    నిర్మాణం ggml_tensor  * a,
    దశాంశం                 c) {
    తిరిగి ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

నిర్మాణం ggml_tensor * ggml_argsort(
        నిర్మాణం ggml_context  * ctx,
        నిర్మాణం ggml_tensor   * a,
        గణన ggml_క్రమీకరించు_క్రమం   క్రమం) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) క్రమం);

    ఫలితం->op     = GGML_OP_ARGSORT;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_argsort_top_k

నిర్మాణం ggml_tensor * ggml_argsort_శిఖరం_k(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_argsort(ctx, a, GGML_క్రమీకరించు_క్రమం_DESC);

    ఫలితం = ggml_view_4d(ctx, ఫలితం,
                k, ఫలితం->ne[1], ఫలితం->ne[2], ఫలితం->ne[3],
                   ఫలితం->nb[1], ఫలితం->nb[2], ఫలితం->nb[3],
                0);

    తిరిగి ఫలితం;
}

// ggml_top_k

నిర్మాణం ggml_tensor * ggml_శిఖరం_k(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    ఫలితం->op     = GGML_OP_శిఖరం_K;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_arange

నిర్మాణం ggml_tensor * ggml_arange(
        నిర్మాణం ggml_context * ctx,
        దశాంశం                 ప్రారంభం,
        దశాంశం                 stop,
        దశాంశం                 అడుగు) {
    GGML_ASSERT(stop > ప్రారంభం);

    స్థిరరాశి int64_t అడుగులు = (int64_t) ceilf((stop - ప్రారంభం) / అడుగు);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, అడుగులు);

    ggml_సమితి_op_params_f32(ఫలితం, 0, ప్రారంభం);
    ggml_సమితి_op_params_f32(ఫలితం, 1, stop);
    ggml_సమితి_op_params_f32(ఫలితం, 2, అడుగు);

    ఫలితం->op = GGML_OP_ARANGE;

    తిరిగి ఫలితం;
}

// ggml_flash_attn_ext

నిర్మాణం ggml_tensor * ggml_flash_attn_ext(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * q,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * ముసుగు,
        దశాంశం                 scale,
        దశాంశం                 max_bias,
        దశాంశం                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    ఒకవేళ (ముసుగు) {
        GGML_ASSERT(ముసుగు->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(ముసుగు));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % ముసుగు->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % ముసుగు->ne[3] == 0);
    }

    ఒకవేళ (max_bias > 0.0f) {
        GGML_ASSERT(ముసుగు);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    దశాంశం params[] = { scale, max_bias, logit_softcap };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_FLASH_ATTN_EXT;
    ఫలితం->src[0] = q;
    ఫలితం->src[1] = k;
    ఫలితం->src[2] = v;
    ఫలితం->src[3] = ముసుగు;

    తిరిగి ఫలితం;
}


శూన్య ggml_flash_attn_ext_సమితి_prec(
        నిర్మాణం ggml_tensor * a,
        గణన ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    స్థిరరాశి int32_t prec_i32 = (int32_t) prec;

    ggml_సమితి_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

గణన ggml_prec ggml_flash_attn_ext_get_prec(
        స్థిరరాశి నిర్మాణం ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    స్థిరరాశి int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    తిరిగి (గణన ggml_prec) prec_i32;
}

శూన్య ggml_flash_attn_ext_సమితి_n_kv_max(
        నిర్మాణం ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_సమితి_op_params_i32(a, 4, n_kv_max);
}

శూన్య ggml_flash_attn_ext_కూడు_sinks(
        నిర్మాణం ggml_tensor * a,
        నిర్మాణం ggml_tensor * sinks) {
    ఒకవేళ (!sinks) {
        a->src[4] = ఖాళీ;
        తిరిగి;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == ఖాళీ);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

నిర్మాణం ggml_tensor * ggml_flash_attn_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * q,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    స్థిరరాశి int64_t     D = q->ne[0];
    స్థిరరాశి int64_t     N = q->ne[1];
    స్థిరరాశి int64_t     M = k->ne[1];
    స్థిరరాశి int64_t   ne2 = q->ne[2];
    స్థిరరాశి int64_t   ne3 = q->ne[3];
    స్థిరరాశి int64_t kvne2 = k->ne[2];

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
    స్థిరరాశి int64_t elem_q = ggml_nelements(q);
    స్థిరరాశి int64_t elem_k = ggml_nelements(k);
    స్థిరరాశి int64_t elem_v = ggml_nelements(v);

    గణన ggml_type ఫలితం_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_కొలత(ఫలితం_type) == 1);
    స్థిరరాశి size_t tsize = ggml_type_కొలత(ఫలితం_type);

    స్థిరరాశి size_t offs_q = 0;
    స్థిరరాశి size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    స్థిరరాశి size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    స్థిరరాశి size_t చివర    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    స్థిరరాశి size_t nelements = (చివర + tsize - 1)/tsize;

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_సమితి_op_params(ఫలితం, &masked_i, పరిమాణం(masked_i));

    ఫలితం->op     = GGML_OP_FLASH_ATTN_వెనుక;
    ఫలితం->src[0] = q;
    ఫలితం->src[1] = k;
    ఫలితం->src[2] = v;
    ఫలితం->src[3] = d;

    తిరిగి ఫలితం;
}

// ggml_ssm_conv

నిర్మాణం ggml_tensor * ggml_ssm_conv(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * sx,
        నిర్మాణం ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_మాతృక(c));

    స్థిరరాశి int64_t d_conv  = c->ne[0];
    స్థిరరాశి int64_t d_inner = c->ne[1];
    స్థిరరాశి int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    స్థిరరాశి int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    ఫలితం->op     = GGML_OP_SSM_CONV;
    ఫలితం->src[0] = sx;
    ఫలితం->src[1] = c;

    తిరిగి ఫలితం;
}

// ggml_ssm_scan

నిర్మాణం ggml_tensor * ggml_ssm_పరిశీలన(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * s,
        నిర్మాణం ggml_tensor  * x,
        నిర్మాణం ggml_tensor  * dt,
        నిర్మాణం ggml_tensor  * A,
        నిర్మాణం ggml_tensor  * B,
        నిర్మాణం ggml_tensor  * C,
        నిర్మాణం ggml_tensor  * ids,
        int64_t               K) {
    GGML_ASSERT(K >= 1);
    GGML_ASSERT(K <= INT32_MAX);
    GGML_ASSERT(ggml_is_contiguous(s));
    GGML_ASSERT(ggml_is_contiguous(dt));
    GGML_ASSERT(ggml_is_contiguous(A));
    GGML_ASSERT(x->nb[0] == ggml_type_కొలత(x->type));
    GGML_ASSERT(B->nb[0] == ggml_type_కొలత(B->type));
    GGML_ASSERT(C->nb[0] == ggml_type_కొలత(C->type));
    GGML_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
    GGML_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
    GGML_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
    GGML_ASSERT(ggml_are_same_shape(B, C));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    {
        స్థిరరాశి int64_t d_state      = s->ne[0];
        స్థిరరాశి int64_t తల_dim     = x->ne[0];
        స్థిరరాశి int64_t n_తల       = x->ne[1];
        స్థిరరాశి int64_t n_seq_టోకెన్లు = x->ne[2];
        స్థిరరాశి int64_t n_seqs       = x->ne[3];

        GGML_ASSERT(dt->ne[0] == n_తల);
        GGML_ASSERT(dt->ne[1] == n_seq_టోకెన్లు);
        GGML_ASSERT(dt->ne[2] == n_seqs);
        GGML_ASSERT(ggml_is_3d(dt));
        GGML_ASSERT(s->ne[1] == తల_dim);
        GGML_ASSERT(s->ne[2] == n_తల);
        GGML_ASSERT(B->ne[0] == d_state);
        GGML_ASSERT(B->ne[2] == n_seq_టోకెన్లు);
        GGML_ASSERT(B->ne[3] == n_seqs);
        GGML_ASSERT(ids->ne[0] == n_seqs);
        GGML_ASSERT(ggml_is_సదిశ(ids));
        GGML_ASSERT(A->ne[1] == n_తల);
        GGML_ASSERT(ggml_is_మాతృక(A));

        ఒకవేళ (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    ఫలితం->op   = GGML_OP_SSM_పరిశీలన;
    ఫలితం->src[0] = s;
    ఫలితం->src[1] = x;
    ఫలితం->src[2] = dt;
    ఫలితం->src[3] = A;
    ఫలితం->src[4] = B;
    ఫలితం->src[5] = C;
    ఫలితం->src[6] = ids;

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) K);

    తిరిగి ఫలితం;
}

// ggml_win_part

నిర్మాణం ggml_tensor * ggml_win_part(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    స్థిరరాశి పూర్ణాంకం px = (w - a->ne[1]%w)%w;
    స్థిరరాశి పూర్ణాంకం py = (w - a->ne[2]%w)%w;

    స్థిరరాశి పూర్ణాంకం npx = (px + a->ne[1])/w;
    స్థిరరాశి పూర్ణాంకం npy = (py + a->ne[2])/w;
    స్థిరరాశి పూర్ణాంకం np  = npx*npy;

    స్థిరరాశి int64_t ne[4] = { a->ne[0], w, w, np, };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_WIN_PART;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_win_unpart

నిర్మాణం ggml_tensor * ggml_win_unpart(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   w0,
        పూర్ణాంకం                   h0,
        పూర్ణాంకం                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_సమితి_op_params(ఫలితం, params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_WIN_UNPART;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_get_rel_pos

నిర్మాణం ggml_tensor * ggml_get_rel_pos(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        పూర్ణాంకం                   qh,
        పూర్ణాంకం                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    స్థిరరాశి int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    ఫలితం->op     = GGML_OP_GET_REL_POS;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

// ggml_add_rel_pos

స్థిర నిర్మాణం ggml_tensor * ggml_కూడు_rel_pos_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * pw,
        నిర్మాణం ggml_tensor  * ph,
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

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_సమితి_op_params_i32(ఫలితం, 0, inplace ? 1 : 0);

    ఫలితం->op     = GGML_OP_కూడు_REL_POS;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = pw;
    ఫలితం->src[2] = ph;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_కూడు_rel_pos(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * pw,
        నిర్మాణం ggml_tensor  * ph) {
    తిరిగి ggml_కూడు_rel_pos_impl(ctx, a, pw, ph, false);
}

నిర్మాణం ggml_tensor * ggml_కూడు_rel_pos_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * pw,
        నిర్మాణం ggml_tensor  * ph) {
    తిరిగి ggml_కూడు_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

నిర్మాణం ggml_tensor * ggml_rwkv_wkv6(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * r,
        నిర్మాణం ggml_tensor  * tf,
        నిర్మాణం ggml_tensor  * td,
        నిర్మాణం ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    స్థిరరాశి int64_t S = k->ne[0];
    స్థిరరాశి int64_t H = k->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = k->ne[2];
    స్థిరరాశి int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    స్థిరరాశి int64_t ne[4] = { S * H, n_టోకెన్లు + S * n_seqs, 1, 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_RWKV_WKV6;
    ఫలితం->src[0] = k;
    ఫలితం->src[1] = v;
    ఫలితం->src[2] = r;
    ఫలితం->src[3] = tf;
    ఫలితం->src[4] = td;
    ఫలితం->src[5] = state;

    తిరిగి ఫలితం;
}

// ggml_gated_linear_attn

నిర్మాణం ggml_tensor * ggml_gated_రేఖీయ_attn(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * q,
        నిర్మాణం ggml_tensor  * g,
        నిర్మాణం ggml_tensor  * state,
        దశాంశం scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    స్థిరరాశి int64_t S = k->ne[0];
    స్థిరరాశి int64_t H = k->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = k->ne[2];
    స్థిరరాశి int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    స్థిరరాశి int64_t ne[4] = { S * H, n_టోకెన్లు + S * n_seqs, 1, 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_సమితి_op_params_f32(ఫలితం, 0, scale);

    ఫలితం->op     = GGML_OP_GATED_రేఖీయ_ATTN;
    ఫలితం->src[0] = k;
    ఫలితం->src[1] = v;
    ఫలితం->src[2] = q;
    ఫలితం->src[3] = g;
    ఫలితం->src[4] = state;

    తిరిగి ఫలితం;
}

// ggml_rwkv_wkv7

నిర్మాణం ggml_tensor * ggml_rwkv_wkv7(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * r,
        నిర్మాణం ggml_tensor  * w,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    స్థిరరాశి int64_t S = k->ne[0];
    స్థిరరాశి int64_t H = k->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = k->ne[2];
    స్థిరరాశి int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    స్థిరరాశి int64_t ne[4] = { S * H, n_టోకెన్లు + S * n_seqs, 1, 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op     = GGML_OP_RWKV_WKV7;
    ఫలితం->src[0] = r;
    ఫలితం->src[1] = w;
    ఫలితం->src[2] = k;
    ఫలితం->src[3] = v;
    ఫలితం->src[4] = a;
    ఫలితం->src[5] = b;
    ఫలితం->src[6] = state;

    తిరిగి ఫలితం;
}

// ggml_unary

స్థిర నిర్మాణం ggml_tensor * ggml_unary_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(a));

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) op);

    ఫలితం->op     = GGML_OP_UNARY;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_unary(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_unary_op    op) {
    తిరిగి ggml_unary_impl(ctx, a, op, false);
}

నిర్మాణం ggml_tensor * ggml_unary_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        గణన ggml_unary_op    op) {
    తిరిగి ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

స్థిర నిర్మాణం ggml_tensor * ggml_map_custom1_impl(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        స్థిరరాశి  ggml_custom1_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    నిర్మాణం ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_MAP_CUSTOM1;
    ఫలితం->src[0] = a;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_map_custom1(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        స్థిరరాశి  ggml_custom1_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

నిర్మాణం ggml_tensor * ggml_map_custom1_inplace(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        స్థిరరాశి  ggml_custom1_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

స్థిర నిర్మాణం ggml_tensor * ggml_map_custom2_impl(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        స్థిరరాశి  ggml_custom2_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    నిర్మాణం ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_MAP_CUSTOM2;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_map_custom2(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        స్థిరరాశి  ggml_custom2_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

నిర్మాణం ggml_tensor * ggml_map_custom2_inplace(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        స్థిరరాశి  ggml_custom2_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

స్థిర నిర్మాణం ggml_tensor * ggml_map_custom3_impl(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        నిర్మాణం ggml_tensor       * c,
        స్థిరరాశి  ggml_custom3_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    నిర్మాణం ggml_tensor * ఫలితం = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    నిర్మాణం ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op     = GGML_OP_MAP_CUSTOM3;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;
    ఫలితం->src[2] = c;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_map_custom3(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        నిర్మాణం ggml_tensor       * c,
        స్థిరరాశి  ggml_custom3_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

నిర్మాణం ggml_tensor * ggml_map_custom3_inplace(
        నిర్మాణం ggml_context      * ctx,
        నిర్మాణం ggml_tensor       * a,
        నిర్మాణం ggml_tensor       * b,
        నిర్మాణం ggml_tensor       * c,
        స్థిరరాశి  ggml_custom3_op_t   fun,
        పూర్ణాంకం                        n_tasks,
        శూన్య                     * userdata) {
    తిరిగి ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

నిర్మాణం ggml_tensor * ggml_custom_4d(
        నిర్మాణం ggml_context * ctx,
        గణన ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        నిర్మాణం ggml_tensor ** వాదనలు_2,
        పూర్ణాంకం                   n_వాదనలు,
        ggml_custom_op_t      fun,
        పూర్ణాంకం                   n_tasks,
        శూన్య                * userdata) {

    GGML_ASSERT(n_వాదనలు < GGML_MAX_SRC);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    నిర్మాణం ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op = GGML_OP_CUSTOM;
    కొరకు (పూర్ణాంకం i = 0; i < n_వాదనలు; i++) {
        ఫలితం->src[i] = వాదనలు_2[i];
    }

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_custom_inplace(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor ** వాదనలు_2,
        పూర్ణాంకం                   n_వాదనలు,
        ggml_custom_op_t      fun,
        పూర్ణాంకం                   n_tasks,
        శూన్య                * userdata) {

    GGML_ASSERT(n_వాదనలు < GGML_MAX_SRC - 1);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);

    నిర్మాణం ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_సమితి_op_params(ఫలితం, &params, పరిమాణం(params));

    ఫలితం->op = GGML_OP_CUSTOM;
    ఫలితం->src[0] = a;
    కొరకు (పూర్ణాంకం i = 0; i < n_వాదనలు; i++) {
        ఫలితం->src[i + 1] = వాదనలు_2[i];
    }

    తిరిగి ఫలితం;
}
// ggml_cross_entropy_loss

నిర్మాణం ggml_tensor * ggml_cross_entropy_loss(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_1d(ctx, a->type, 1);

    ఫలితం->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_cross_entropy_loss_back

నిర్మాణం ggml_tensor * ggml_cross_entropy_loss_వెనుక(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        నిర్మాణం ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    నిర్మాణం ggml_tensor * ఫలితం = ggml_dup_tensor(ctx, b);

    ఫలితం->op     = GGML_OP_CROSS_ENTROPY_LOSS_వెనుక;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;
    ఫలితం->src[2] = c;

    తిరిగి ఫలితం;
}

// opt_step_adamw

నిర్మాణం ggml_tensor * ggml_opt_అడుగు_adamw(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * grad,
        నిర్మాణం ggml_tensor  * m,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_OPT_అడుగు_ADAMW;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = grad;
    ఫలితం->src[2] = m;
    ఫలితం->src[3] = v;
    ఫలితం->src[4] = adamw_params;

    తిరిగి ఫలితం;
}

// opt_step_sgd

నిర్మాణం ggml_tensor * ggml_opt_అడుగు_sgd(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * grad,
        నిర్మాణం ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_view_tensor(ctx, a);

    ఫలితం->op     = GGML_OP_OPT_అడుగు_SGD;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = grad;
    ఫలితం->src[2] = params;

    తిరిగి ఫలితం;
}

// solve_tri

నిర్మాణం ggml_tensor * ggml_solve_tri(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * a,
        నిర్మాణం ggml_tensor  * b,
        bool                  ఎడమ_2,
        bool                  క్రింది,
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

    GGML_ASSERT(క్రింది && ఎడమ_2 && !uni); // TODO: support other variants

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    ఫలితం->op     = GGML_OP_SOLVE_TRI;
    ఫలితం->src[0] = a;
    ఫలితం->src[1] = b;

    తిరిగి ఫలితం;
}

// ggml_gated_delta_net

నిర్మాణం ggml_tensor * ggml_gated_delta_net(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * q,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * v,
        నిర్మాణం ggml_tensor  * g,
        నిర్మాణం ggml_tensor  * beta,
        నిర్మాణం ggml_tensor  * state,
        int64_t               K) {
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(q));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(k));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(v));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(beta));
    GGML_ASSERT(ggml_is_contiguous(state));

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(beta->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t S_v      = v->ne[0];
    స్థిరరాశి int64_t H        = v->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = v->ne[2];
    స్థిరరాశి int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    స్థిరరాశి int64_t state_అడ్డువరుసలు = K * S_v * n_seqs;
    స్థిరరాశి int64_t ne[4] = { S_v * H, n_టోకెన్లు * n_seqs + state_అడ్డువరుసలు, 1, 1 };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_సమితి_op_params_i32(ఫలితం, 0, (int32_t) K);

    ఫలితం->op     = GGML_OP_GATED_DELTA_NET;
    ఫలితం->src[0] = q;
    ఫలితం->src[1] = k;
    ఫలితం->src[2] = v;
    ఫలితం->src[3] = g;
    ఫలితం->src[4] = beta;
    ఫలితం->src[5] = state;

    తిరిగి ఫలితం;
}

// ggml_lightning_indexer

నిర్మాణం ggml_tensor * ggml_lightning_indexer(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * q,
        నిర్మాణం ggml_tensor  * k,
        నిర్మాణం ggml_tensor  * weights,
        నిర్మాణం ggml_tensor  * ముసుగు) {

    GGML_ASSERT(       q->type == GGML_TYPE_F32);
    GGML_ASSERT( weights->type == GGML_TYPE_F32);
    GGML_ASSERT(    ముసుగు->type == GGML_TYPE_F16);
    GGML_ASSERT(      q->ne[0] == k->ne[0]);
    GGML_ASSERT(   ముసుగు->ne[0] == k->ne[2]);
    GGML_ASSERT(      q->ne[1] == weights->ne[0]);
    GGML_ASSERT(      k->ne[1] == 1);
    GGML_ASSERT(   ముసుగు->ne[1] == q->ne[2]);
    GGML_ASSERT(      q->ne[2] == weights->ne[1]);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(   ముసుగు->ne[2] == 1);
    GGML_ASSERT(      q->ne[3] == k->ne[3]);
    GGML_ASSERT(      k->ne[3] == weights->ne[3]);
    GGML_ASSERT(weights->ne[3] % ముసుగు->ne[3] == 0);

    int64_t ne[4] = { k->ne[2], q->ne[2], 1, q->ne[3] };
    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ఫలితం->op   = GGML_OP_LIGHTNING_INDEXER;
    ఫలితం->src[0] = q;
    ఫలితం->src[1] = k;
    ఫలితం->src[2] = weights;
    ఫలితం->src[3] = ముసుగు;

    తిరిగి ఫలితం;
}

// ggml_dsv4_hc_comb

నిర్మాణం ggml_tensor * ggml_dsv4_hc_comb(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * mixes,
        నిర్మాణం ggml_tensor  * scale,
        నిర్మాణం ggml_tensor  * పునాది,
        దశాంశం                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(పునాది->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    స్థిరరాశి int64_t hc_mix_dim = mixes->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు   = mixes->ne[1];

    int64_t hc = 0;
    కొరకు (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        ఒకవేళ ((2 + i)*i == hc_mix_dim) {
            hc = i;
            ఆపు;
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
    GGML_ASSERT(పునాది->ne[0] == hc_mix_dim);
    GGML_ASSERT(పునాది->ne[1] == 1);
    GGML_ASSERT(పునాది->ne[2] == 1);
    GGML_ASSERT(పునాది->ne[3] == 1);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_టోకెన్లు);

    ggml_సమితి_op_params_f32(ఫలితం, 0, eps);
    ggml_సమితి_op_params_i32(ఫలితం, 1, n_iter);

    ఫలితం->op     = GGML_OP_DSV4_HC_COMB;
    ఫలితం->src[0] = mixes;
    ఫలితం->src[1] = scale;
    ఫలితం->src[2] = పునాది;

    తిరిగి ఫలితం;
}

// ggml_dsv4_hc_pre

స్థిర నిర్మాణం ggml_tensor * ggml_dsv4_hc_pre_impl(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * x,
        నిర్మాణం ggml_tensor  * weights,
        దశాంశం                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t hc       = x->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    ఒకవేళ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_టోకెన్లు);
    } లేకపోతే {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_టోకెన్లు);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_టోకెన్లు);

    ggml_సమితి_op_params_f32(ఫలితం, 0, scale);
    ggml_సమితి_op_params_i32(ఫలితం, 1, gated ? 1 : 0);

    ఫలితం->op     = GGML_OP_DSV4_HC_PRE;
    ఫలితం->src[0] = x;
    ఫలితం->src[1] = weights;

    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_dsv4_hc_pre(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * x,
        నిర్మాణం ggml_tensor  * weights) {
    తిరిగి ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

నిర్మాణం ggml_tensor * ggml_dsv4_hc_pre_gated(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * x,
        నిర్మాణం ggml_tensor  * gate,
        దశాంశం                 scale) {
    తిరిగి ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

నిర్మాణం ggml_tensor * ggml_dsv4_hc_post(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * x,
        నిర్మాణం ggml_tensor  * residual,
        నిర్మాణం ggml_tensor  * post,
        నిర్మాణం ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[1];
    స్థిరరాశి int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[2] == 1);
    GGML_ASSERT(x->ne[3] == 1);

    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_టోకెన్లు);
    GGML_ASSERT(residual->ne[3] == 1);

    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_టోకెన్లు);
    GGML_ASSERT(post->ne[2] == 1);
    GGML_ASSERT(post->ne[3] == 1);

    ఒకవేళ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_టోకెన్లు);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_టోకెన్లు);

    ఫలితం->op     = GGML_OP_DSV4_HC_POST;
    ఫలితం->src[0] = x;
    ఫలితం->src[1] = residual;
    ఫలితం->src[2] = post;
    ఫలితం->src[3] = comb;

    తిరిగి ఫలితం;
}

// ggml_top_k_mask

నిర్మాణం ggml_tensor * ggml_శిఖరం_k_ముసుగు(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_tensor  * ముసుగు,
        నిర్మాణం ggml_tensor  * శిఖరం_k,
        నిర్మాణం ggml_tensor  * score,
        నిర్మాణం ggml_tensor  * ఖండాలు,
        నిర్మాణం ggml_tensor  * extra) {
    GGML_ASSERT(ముసుగు->type == GGML_TYPE_F16 || ముసుగు->type == GGML_TYPE_F32);
    GGML_ASSERT(ముసుగు->nb[0] == ggml_type_కొలత(ముసుగు->type));
    GGML_ASSERT(శిఖరం_k->type == GGML_TYPE_I32);
    GGML_ASSERT(శిఖరం_k->ne[1] % ముసుగు->ne[1] == 0 && శిఖరం_k->ne[2] % ముసుగు->ne[2] == 0 && శిఖరం_k->ne[3] % ముసుగు->ne[3] == 0);

    ఒకవేళ (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == శిఖరం_k->ne[1] && score->ne[2] == శిఖరం_k->ne[2] && score->ne[3] == శిఖరం_k->ne[3]);
    }

    ఒకవేళ (ఖండాలు) {
        GGML_ASSERT(ఖండాలు->type == GGML_TYPE_I32);
        GGML_ASSERT(ఖండాలు->ne[2] == 1 && ఖండాలు->ne[3] == 1);
        GGML_ASSERT(!score || ఖండాలు->ne[1] == score->ne[0]);
    }

    ఒకవేళ (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == శిఖరం_k->ne[1] && extra->ne[2] == శిఖరం_k->ne[2] && extra->ne[3] == శిఖరం_k->ne[3]);
    }

    నిర్మాణం ggml_tensor * ఫలితం = ggml_new_tensor_4d(ctx, ముసుగు->type, ముసుగు->ne[0], శిఖరం_k->ne[1], శిఖరం_k->ne[2], శిఖరం_k->ne[3]);

    ఫలితం->op     = GGML_OP_శిఖరం_K_ముసుగు;
    ఫలితం->src[0] = ముసుగు;
    ఫలితం->src[1] = శిఖరం_k;
    ఫలితం->src[2] = score;
    ఫలితం->src[3] = ఖండాలు;
    ఫలితం->src[4] = extra;

    తిరిగి ఫలితం;
}

////////////////////////////////////////////////////////////////////////////////

నిర్మాణం ggml_hash_సమితి ggml_hash_సమితి_new(size_t కొలత) {
    కొలత = ggml_hash_కొలత(కొలత);
    నిర్మాణం ggml_hash_సమితి ఫలితం;
    ఫలితం.కొలత = కొలత;
    ఫలితం.కీలకాలు = GGML_MALLOC(పరిమాణం(నిర్మాణం ggml_tensor *) * కొలత);
    ఫలితం.వాడిన = GGML_CALLOC(ggml_bitset_కొలత(కొలత), పరిమాణం(ggml_bitset_t));
    తిరిగి ఫలితం;
}

శూన్య ggml_hash_సమితి_reset(నిర్మాణం ggml_hash_సమితి * hash_సమితి) {
    memset(hash_సమితి->వాడిన, 0, పరిమాణం(ggml_bitset_t) * ggml_bitset_కొలత(hash_సమితి->కొలత));
}

శూన్య ggml_hash_సమితి_free(నిర్మాణం ggml_hash_సమితి * hash_సమితి) {
    GGML_FREE(hash_సమితి->వాడిన);
    GGML_FREE(hash_సమితి->కీలకాలు);
}

size_t ggml_hash_కొలత(size_t min_sz) {
    // next primes after powers of two
    స్థిర స్థిరరాశి size_t ప్రధానసంఖ్యలు[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    స్థిర స్థిరరాశి size_t n_ప్రధానసంఖ్యలు = పరిమాణం(ప్రధానసంఖ్యలు)/పరిమాణం(ప్రధానసంఖ్యలు[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_ప్రధానసంఖ్యలు;
    అయితే (l < r) {
        size_t m = (l + r)/2;
        ఒకవేళ (ప్రధానసంఖ్యలు[m] < min_sz) {
            l = m + 1;
        } లేకపోతే {
            r = m;
        }
    }
    size_t sz = l < n_ప్రధానసంఖ్యలు ? ప్రధానసంఖ్యలు[l] : min_sz | 1;
    తిరిగి sz;
}

నిర్మాణం hash_map {
    నిర్మాణం ggml_hash_సమితి సమితి;
    నిర్మాణం ggml_tensor ** vals;
};

స్థిర నిర్మాణం hash_map * ggml_new_hash_map(size_t కొలత) {
    నిర్మాణం hash_map * ఫలితం = GGML_MALLOC(పరిమాణం(నిర్మాణం hash_map));
    ఫలితం->సమితి = ggml_hash_సమితి_new(కొలత);
    ఫలితం->vals = GGML_CALLOC(ఫలితం->సమితి.కొలత, పరిమాణం(నిర్మాణం ggml_tensor *));
    తిరిగి ఫలితం;
}

స్థిర శూన్య ggml_hash_map_free(నిర్మాణం hash_map * map) {
    ggml_hash_సమితి_free(&map->సమితి);
    GGML_FREE(map->vals);
    GGML_FREE(map);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

స్థిర శూన్య ggml_కూడు_or_సమితి(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_cgraph  * cgraph,
        size_t                isrc,
        నిర్మాణం ggml_tensor  * tensor) {
    నిర్మాణం ggml_tensor * src = cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[isrc];
    GGML_ASSERT(src);
    ఒకవేళ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_కూడు_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } లేకపోతే {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

స్థిర శూన్య ggml_acc_or_సమితి(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_cgraph  * cgraph,
        size_t                isrc,
        నిర్మాణం ggml_tensor  * tensor,
        స్థిరరాశి  size_t         nb1,
        స్థిరరాశి  size_t         nb2,
        స్థిరరాశి  size_t         nb3,
        స్థిరరాశి  size_t         స్థానభ్రంశం) {
    నిర్మాణం ggml_tensor * src = cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[isrc];
    GGML_ASSERT(src);
    ఒకవేళ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, స్థానభ్రంశం, cgraph->grad_accs[isrc]);
    } లేకపోతే {
        నిర్మాణం ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, స్థానభ్రంశం, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

స్థిర శూన్య ggml_add1_or_సమితి(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_cgraph  * cgraph,
        size_t                isrc,
        నిర్మాణం ggml_tensor  * tensor) {
    నిర్మాణం ggml_tensor * src = cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[isrc];
    GGML_ASSERT(src);
    ఒకవేళ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } లేకపోతే {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

స్థిర శూన్య ggml_sub_or_సమితి(
        నిర్మాణం ggml_context * ctx,
        నిర్మాణం ggml_cgraph  * cgraph,
        size_t                isrc,
        నిర్మాణం ggml_tensor  * tensor) {
    నిర్మాణం ggml_tensor * src = cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[isrc];
    GGML_ASSERT(src);
    ఒకవేళ (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } లేకపోతే {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

స్థిర శూన్య ggml_compute_backward(
        నిర్మాణం ggml_context * ctx, నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం i, స్థిరరాశి bool * grads_needed) {
    నిర్మాణం ggml_tensor * tensor = cgraph->nodes[i];
    నిర్మాణం ggml_tensor * grad   = ggml_లేఖాచిత్రం_get_grad(cgraph, tensor);

    ఒకవేళ (!grad) {
        తిరిగి;
    }

    నిర్మాణం ggml_tensor * src0 = tensor->src[0];
    నిర్మాణం ggml_tensor * src1 = tensor->src[1];
    నిర్మాణం ggml_tensor * src2 = tensor->src[2];
    నిర్మాణం ggml_hash_సమితి * hash_సమితి = &cgraph->సందర్శించిన_hash_సమితి;
    స్థిరరాశి size_t isrc0 = src0 ? ggml_hash_కనుగొను(hash_సమితి, src0) : (size_t) -1;
    స్థిరరాశి size_t isrc1 = src1 ? ggml_hash_కనుగొను(hash_సమితి, src1) : (size_t) -1;
    స్థిరరాశి size_t isrc2 = src2 ? ggml_hash_కనుగొను(hash_సమితి, src2) : (size_t) -1;
    స్థిరరాశి bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_నిండిన && ggml_bitset_get(hash_సమితి->వాడిన, isrc0) && grads_needed[isrc0];
    స్థిరరాశి bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_నిండిన && ggml_bitset_get(hash_సమితి->వాడిన, isrc1) && grads_needed[isrc1];
    స్థిరరాశి bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_నిండిన && ggml_bitset_get(hash_సమితి->వాడిన, isrc2) && grads_needed[isrc2];

    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_DUP: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, grad);
            }
        } ఆపు;
        స్థితి GGML_OP_కూడు: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, grad);
            }
            ఒకవేళ (src1_needs_grads) {
                నిర్మాణం ggml_tensor * tmp = grad;
                ఒకవేళ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_వెనుక(ctx, tmp, src1);
                }
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, tmp);
            }
        } ఆపు;
        స్థితి GGML_OP_ADD1: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, grad);
            }
            ఒకవేళ (src1_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, ggml_సగటు(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } ఆపు;
        స్థితి GGML_OP_ACC: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, grad);
            }
            ఒకవేళ (src1_needs_grads) {
                స్థిరరాశి size_t nb1    = ((int32_t *) tensor->op_params)[0];
                స్థిరరాశి size_t nb2    = ((int32_t *) tensor->op_params)[1];
                స్థిరరాశి size_t nb3    = ((int32_t *) tensor->op_params)[2];
                స్థిరరాశి size_t స్థానభ్రంశం = ((int32_t *) tensor->op_params)[3];

                నిర్మాణం ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, స్థానభ్రంశం);

                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ఆపు;
        స్థితి GGML_OP_SUB: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, grad);
            }
            ఒకవేళ (src1_needs_grads) {
                ggml_sub_or_సమితి(ctx, cgraph, isrc1, grad);
            }
        } ఆపు;
        స్థితి GGML_OP_MUL: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            ఒకవేళ (src1_needs_grads) {
                నిర్మాణం ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                ఒకవేళ (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_వెనుక(ctx, tmp, src1);
                }
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, tmp);
            }
        } ఆపు;
        స్థితి GGML_OP_DIV: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            ఒకవేళ (src1_needs_grads) {
                ggml_sub_or_సమితి(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } ఆపు;
        స్థితి GGML_OP_SQR: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } ఆపు;
        స్థితి GGML_OP_SQRT: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } ఆపు;
        స్థితి GGML_OP_LOG: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_SIN: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } ఆపు;
        స్థితి GGML_OP_COS: {
            ఒకవేళ (src0_needs_grads) {
                ggml_sub_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } ఆపు;
        స్థితి GGML_OP_కూడిక: {
            ఒకవేళ (src0_needs_grads) {
                ggml_add1_or_సమితి(ctx, cgraph, isrc0, grad);
            }
        } ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_సగటు: {
            ఒకవేళ (src0_needs_grads) {
                ggml_add1_or_సమితి(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } ఆపు;
        స్థితి GGML_OP_REPEAT: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_repeat_వెనుక(ctx, grad, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_REPEAT_వెనుక: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_RMS_NORM: {
            ఒకవేళ (src0_needs_grads) {
                దశాంశం eps;
                జ్ఞాపకం_నకలు(&eps, tensor->op_params, పరిమాణం(దశాంశం));
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_rms_norm_వెనుక(ctx, grad, src0, eps));
            }
        } ఆపు;
        స్థితి GGML_OP_MUL_MAT: {
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

            ఒకవేళ (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                నిర్మాణం ggml_tensor * tmp =
                    ggml_బహిర్గతం_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                ఒకవేళ (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    స్థిరరాశి int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    స్థిరరాశి size_t nb2 = tmp->nb[2] * nr2;
                    స్థిరరాశి size_t nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_వెనుక(ctx, tmp, src0);
                }
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, tmp);
            }
            ఒకవేళ (src1_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1,
                        // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                        //     ggml_cont(ctx,                  // [m,n,q1,r1]
                        //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                        //     grad),                          // [m,p,qq,rr]

                        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                        // avoid transpose of src0, rather transpose smaller tensor->grad
                        // and then use ggml_out_prod
                        ggml_బహిర్గతం_prod(ctx,      // [n,p,qq,rr]
                            src0,               // [n,m,q1,r1]
                            ggml_transpose(ctx, // [p,m,qq,rr]
                                grad)));        // [m,p,qq,rr]
            }
        } ఆపు;
        స్థితి GGML_OP_SCALE: {
            ఒకవేళ (src0_needs_grads) {
                దశాంశం s;
                జ్ఞాపకం_నకలు(&s, tensor->op_params, పరిమాణం(దశాంశం));
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } ఆపు;
        స్థితి GGML_OP_సమితి: {
            స్థిరరాశి size_t nb1    = ((స్థిరరాశి int32_t *) tensor->op_params)[0];
            స్థిరరాశి size_t nb2    = ((స్థిరరాశి int32_t *) tensor->op_params)[1];
            స్థిరరాశి size_t nb3    = ((స్థిరరాశి int32_t *) tensor->op_params)[2];
            స్థిరరాశి size_t స్థానభ్రంశం = ((స్థిరరాశి int32_t *) tensor->op_params)[3];

            నిర్మాణం ggml_tensor * tensor_grad_view = ఖాళీ;

            ఒకవేళ (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, స్థానభ్రంశం);
            }

            ఒకవేళ (src0_needs_grads) {
                నిర్మాణం ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, స్థానభ్రంశం, false));
            }

            ఒకవేళ (src1_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } ఆపు;
        స్థితి GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            ఒకవేళ (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            ఒకవేళ (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } ఆపు;
        స్థితి GGML_OP_CONT: {
            // same as cpy
            ఒకవేళ (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_RESHAPE: {
            ఒకవేళ (src0_needs_grads) {
                నిర్మాణం ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } ఆపు;
        స్థితి GGML_OP_VIEW: {
            ఒకవేళ (src0_needs_grads) {
                size_t స్థానభ్రంశం;

                జ్ఞాపకం_నకలు(&స్థానభ్రంశం, tensor->op_params, పరిమాణం(స్థానభ్రంశం));

                size_t nb1 = tensor->nb[1];
                size_t nb2 = tensor->nb[2];
                size_t nb3 = tensor->nb[3];

                ఒకవేళ (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    size_t ng = ggml_element_కొలత(cgraph->grads[isrc0]);
                    size_t n0 = ggml_element_కొలత(src0);
                    GGML_ASSERT(స్థానభ్రంశం % n0 == 0);
                    GGML_ASSERT(nb1 % n0 == 0);
                    GGML_ASSERT(nb2 % n0 == 0);
                    GGML_ASSERT(nb3 % n0 == 0);
                    స్థానభ్రంశం = (స్థానభ్రంశం / n0) * ng;
                    nb1 = (nb1 / n0) * ng;
                    nb2 = (nb2 / n0) * ng;
                    nb3 = (nb3 / n0) * ng;
                }

                ggml_acc_or_సమితి(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, స్థానభ్రంశం);
            }
        } ఆపు;
        స్థితి GGML_OP_క్రమమార్పు: {
            ఒకవేళ (src0_needs_grads) {
                స్థిరరాశి int32_t * axes = (స్థిరరాశి int32_t *) tensor->op_params;
                స్థిరరాశి పూర్ణాంకం axis0 = axes[0] & 0x3;
                స్థిరరాశి పూర్ణాంకం axis1 = axes[1] & 0x3;
                స్థిరరాశి పూర్ణాంకం axis2 = axes[2] & 0x3;
                స్థిరరాశి పూర్ణాంకం axis3 = axes[3] & 0x3;
                పూర్ణాంకం axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_క్రమమార్పు(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } ఆపు;
        స్థితి GGML_OP_TRANSPOSE: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_get_అడ్డువరుసలు_వెనుక(ctx, grad, src1, src0));
            }
            ఒకవేళ (src1_needs_grads) {
                // noop
            }
        } ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_INF: {
            ఒకవేళ (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                స్థిరరాశి పూర్ణాంకం n_past = ((స్థిరరాశి int32_t *) tensor->op_params)[0];
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_diag_ముసుగు_zero_impl(ctx, grad, n_past, false));
            }
        } ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_ZERO: {
            ఒకవేళ (src0_needs_grads) {
                స్థిరరాశి పూర్ణాంకం n_past = ((స్థిరరాశి int32_t *) tensor->op_params)[0];
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_diag_ముసుగు_zero_impl(ctx, grad, n_past, false));
            }
        } ఆపు;
        స్థితి GGML_OP_SOFT_MAX: {
            ఒకవేళ (src0_needs_grads) {
                దశాంశం scale    = 1.0f;
                దశాంశం max_bias = 0.0f;

                జ్ఞాపకం_నకలు(&scale,    (స్థిరరాశి దశాంశం *) tensor->op_params + 0, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) tensor->op_params + 1, పరిమాణం(దశాంశం));

                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_soft_max_ext_వెనుక(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } ఆపు;
        స్థితి GGML_OP_ROPE: {
            ఒకవేళ (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                స్థిరరాశి పూర్ణాంకం n_dims     = ((స్థిరరాశి int32_t *) tensor->op_params)[1];
                స్థిరరాశి పూర్ణాంకం బహుళకం       = ((స్థిరరాశి int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                స్థిరరాశి పూర్ణాంకం n_ctx_orig = ((స్థిరరాశి int32_t *) tensor->op_params)[4];
                దశాంశం freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow;
                పూర్ణాంకం sections[4] = {0, 0, 0, 0};

                జ్ఞాపకం_నకలు(&freq_పునాది,   (స్థిరరాశి దశాంశం *) tensor->op_params +  5, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&freq_scale,  (స్థిరరాశి దశాంశం *) tensor->op_params +  6, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&ext_గుణకం,  (స్థిరరాశి దశాంశం *) tensor->op_params +  7, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&attn_గుణకం, (స్థిరరాశి దశాంశం *) tensor->op_params +  8, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&beta_fast,   (స్థిరరాశి దశాంశం *) tensor->op_params +  9, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&beta_slow,   (స్థిరరాశి దశాంశం *) tensor->op_params + 10, పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(&sections,                    tensor->op_params + 11, పరిమాణం(sections));

                నిర్మాణం ggml_tensor * rope_వెనుక = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_వెనుక(ctx, grad, src1, src2, n_dims,
                        బహుళకం, n_ctx_orig, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow) :
                    ggml_rope_multi_వెనుక(ctx, grad, src1, src2, n_dims, sections,
                        బహుళకం, n_ctx_orig, freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow);
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, rope_వెనుక);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } ఆపు;
        స్థితి GGML_OP_IM2_నిలువు: {
            ఒకవేళ (src1_needs_grads) {
                స్థిరరాశి int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                స్థిరరాశి int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                స్థిరరాశి int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                స్థిరరాశి int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                స్థిరరాశి int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                స్థిరరాశి int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                స్థిరరాశి bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, ggml_im2col_వెనుక(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } ఆపు;
        స్థితి GGML_OP_POOL_2D: {
            ఒకవేళ (src0_needs_grads) {
                స్థిరరాశి గణన ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                స్థిరరాశి      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                స్థిరరాశి      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                స్థిరరాశి      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                స్థిరరాశి      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                స్థిరరాశి      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                స్థిరరాశి      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_pool_2d_వెనుక(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } ఆపు;
        స్థితి GGML_OP_WIN_PART:
        స్థితి GGML_OP_WIN_UNPART:
        స్థితి GGML_OP_UNARY: {
            ఎంపిక (ggml_get_unary_op(tensor)) {
                స్థితి GGML_UNARY_OP_ABS: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_SGN: {
                    // noop
                } ఆపు;
                స్థితి GGML_UNARY_OP_NEG: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_sub_or_సమితి(ctx, cgraph, isrc0, grad);
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_అడుగు: {
                    // noop
                } ఆపు;
                స్థితి GGML_UNARY_OP_RELU: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_అడుగు(ctx, src0), grad));
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_SILU: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_silu_వెనుక(ctx, grad, src0));
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_EXP: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_EXPM1: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } ఆపు;
                స్థితి GGML_UNARY_OP_SOFTPLUS: {
                    ఒకవేళ (src0_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } ఆపు;
                అప్రమేయ: {
                    fprintf(stderr, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS: {
            ఒకవేళ (src0_needs_grads) {
                ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_cross_entropy_loss_వెనుక(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } ఆపు;
        స్థితి GGML_OP_GLU: {
            ఎంపిక (ggml_get_glu_op(tensor)) {
                స్థితి GGML_GLU_OP_SWIGLU: {
                    ఒకవేళ (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc0, ggml_silu_వెనుక(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    ఒకవేళ (src1_needs_grads) {
                        ggml_కూడు_or_సమితి(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } ఆపు;
                అప్రమేయ: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } ఆపు;
        స్థితి GGML_OP_NONE: {
            // noop
        } ఆపు;
        స్థితి GGML_OP_లెక్క:
        అప్రమేయ: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

స్థిర size_t ggml_సందర్శించు_parents_లేఖాచిత్రం(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_tensor * కణుపు, bool compute) {
    ఒకవేళ (కణుపు->op != GGML_OP_NONE && compute) {
        కణుపు->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    స్థిరరాశి size_t కణుపు_hash_pos = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు);
    GGML_ASSERT(కణుపు_hash_pos != GGML_HASHSET_నిండిన);

    ఒకవేళ (ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, కణుపు_hash_pos)) {
        // already visited

        ఒకవేళ (compute) {
            // update the compute flag regardless
            కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; ++i) {
                నిర్మాణం ggml_tensor * src = కణుపు->src[i];
                ఒకవేళ (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_సందర్శించు_parents_లేఖాచిత్రం(cgraph, src, true);
                }
            }
        }

        తిరిగి కణుపు_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->సందర్శించిన_hash_సమితి.కీలకాలు[కణుపు_hash_pos] = కణుపు;
    ggml_bitset_సమితి(cgraph->సందర్శించిన_hash_సమితి.వాడిన, కణుపు_hash_pos);
    cgraph->use_counts[కణుపు_hash_pos] = 0;

    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; ++i) {
        స్థిరరాశి పూర్ణాంకం k =
            (cgraph->క్రమం == GGML_CGRAPH_EVAL_క్రమం_ఎడమ_గమ్యం_కుడి) ? i :
            (cgraph->క్రమం == GGML_CGRAPH_EVAL_క్రమం_కుడి_గమ్యం_ఎడమ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        నిర్మాణం ggml_tensor * src = కణుపు->src[k];
        ఒకవేళ (src) {
            స్థిరరాశి size_t src_hash_pos = ggml_సందర్శించు_parents_లేఖాచిత్రం(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    ఒకవేళ (కణుపు->op == GGML_OP_NONE && !(కణుపు->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->కొలత);

        ఒకవేళ (అక్షరశ్రేణి_పొడవు(కణుపు->name) == 0) {
            ggml_format_name(కణుపు, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = కణుపు;
        cgraph->n_leafs++;
    } లేకపోతే {
        GGML_ASSERT(cgraph->n_nodes < cgraph->కొలత);

        ఒకవేళ (అక్షరశ్రేణి_పొడవు(కణుపు->name) == 0) {
            ggml_format_name(కణుపు, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = కణుపు;
        cgraph->n_nodes++;
    }

    తిరిగి కణుపు_hash_pos;
}

స్థిర శూన్య ggml_build_forward_impl(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_tensor * tensor, bool expand, bool compute) {
    ఒకవేళ (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_లేఖాచిత్రం_clear(cgraph);
    }

    స్థిరరాశి పూర్ణాంకం n_old = cgraph->n_nodes;

    ggml_సందర్శించు_parents_లేఖాచిత్రం(cgraph, tensor, compute);

    స్థిరరాశి పూర్ణాంకం n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    ఒకవేళ (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

నిర్మాణం ggml_tensor * ggml_build_forward_select(
        నిర్మాణం ggml_cgraph  * cgraph,
        నిర్మాణం ggml_tensor ** tensors,
        పూర్ణాంకం                   n_tensors,
        పూర్ణాంకం                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    కొరకు (పూర్ణాంకం i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    తిరిగి tensors[idx];
}

శూన్య ggml_build_forward_expand(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

శూన్య ggml_build_forward_క్రమం(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

శూన్య ggml_build_backward_expand(
        నిర్మాణం ggml_context *  ctx,
        నిర్మాణం ggml_cgraph  *  cgraph,
        నిర్మాణం ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    స్థిరరాశి పూర్ణాంకం n_nodes_f = cgraph->n_nodes;

    memset(cgraph->grads,     0, cgraph->సందర్శించిన_hash_సమితి.కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
    memset(cgraph->grad_accs, 0, cgraph->సందర్శించిన_hash_సమితి.కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
    bool * grads_needed = సున్నా_జ్ఞాపకం_ఇవ్వు(cgraph->సందర్శించిన_hash_సమితి.కొలత, పరిమాణం(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        కొరకు (పూర్ణాంకం i = 0; i < n_nodes_f; ++i) {
            నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];
            any_params = any_params || (కణుపు->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (కణుపు->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    కొరకు (పూర్ణాంకం i = 0; i < n_nodes_f; ++i) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        ఒకవేళ (కణుపు->type == GGML_TYPE_I32) {
            కొనసాగు;
        }

        bool కణుపు_needs_grad = (కణుపు->flags & GGML_TENSOR_FLAG_PARAM) || (కణుపు->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        ఎంపిక (కణుపు->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            స్థితి GGML_OP_IM2_నిలువు:      // only used for its shape
            స్థితి GGML_OP_IM2_నిలువు_వెనుక: // same as IM2COL
                ignore_src[0] = true;
                ఆపు;
            స్థితి GGML_OP_UNARY: {
                స్థిరరాశి గణన ggml_unary_op uop = ggml_get_unary_op(కణుపు);
                // SGN and STEP unary ops are piecewise constant
                ఒకవేళ (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_అడుగు) {
                    ignore_src[0] = true;
                }
            } ఆపు;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            స్థితి GGML_OP_CPY:           // gradients in CPY target are irrelevant
            స్థితి GGML_OP_GET_అడ్డువరుసలు:      // row indices not differentiable
            స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక: // same as for GET_ROWS
            స్థితి GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                ఆపు;

            అప్రమేయ:
                ఆపు;
        }
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; ++j) {
            ఒకవేళ (!కణుపు->src[j] || ignore_src[j] || !grads_needed[ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు->src[j])]) {
                కొనసాగు;
            }
            GGML_ASSERT(కణుపు->src[j]->type == GGML_TYPE_F32 || కణుపు->src[j]->type == GGML_TYPE_F16);
            కణుపు_needs_grad = true;
            ఆపు;
        }
        ఒకవేళ (!కణుపు_needs_grad) {
            కొనసాగు;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!కణుపు->view_src || కణుపు->op == GGML_OP_CPY || కణుపు->op == GGML_OP_సమితి_అడ్డువరుసలు || కణుపు->op == GGML_OP_VIEW ||
            కణుపు->op == GGML_OP_RESHAPE || కణుపు->op == GGML_OP_క్రమమార్పు || కణుపు->op == GGML_OP_TRANSPOSE);

        స్థిరరాశి size_t ihash = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు);
        GGML_ASSERT(ihash != GGML_HASHSET_నిండిన);
        GGML_ASSERT(ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, ihash));
        ఒకవేళ (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } లేకపోతే ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, కణుపు->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    కొరకు (పూర్ణాంకం i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    విడుదల(grads_needed);
}

స్థిర శూన్య * incr_ptr_aligned(శూన్య ** p, size_t కొలత, size_t align) {
    శూన్య * ptr = *p;
    ptr = (శూన్య *) GGML_PAD((uintptr_t) ptr, align);
    *p = (శూన్య *) ((అక్షరం *) ptr + కొలత);
    తిరిగి ptr;
}

స్థిర size_t ggml_లేఖాచిత్రం_nbytes(size_t కొలత, bool grads) {
    size_t hash_కొలత = ggml_hash_కొలత(కొలత * 2);
    శూన్య * p = (అక్షరం *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, పరిమాణం(నిర్మాణం ggml_cgraph), 1);
    incr_ptr_aligned(&p, కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(int32_t), పరిమాణం(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)); // hash keys
    ఒకవేళ (grads) {
        incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_కొలత(hash_కొలత) * పరిమాణం(ggml_bitset_t), పరిమాణం(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    తిరిగి nbytes - 1024;
}

size_t ggml_లేఖాచిత్రం_overhead_custom(size_t కొలత, bool grads) {
    తిరిగి GGML_OBJECT_కొలత + GGML_PAD(ggml_లేఖాచిత్రం_nbytes(కొలత, grads), GGML_MEM_ALIGN);
}

size_t ggml_లేఖాచిత్రం_overhead(శూన్య) {
    తిరిగి ggml_లేఖాచిత్రం_overhead_custom(GGML_DEFAULT_లేఖాచిత్రం_కొలత, false);
}

నిర్మాణం ggml_cgraph * ggml_new_లేఖాచిత్రం_custom(నిర్మాణం ggml_context * ctx, size_t కొలత, bool grads) {
    స్థిరరాశి size_t obj_కొలత = ggml_లేఖాచిత్రం_nbytes(కొలత, grads);
    నిర్మాణం ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_లేఖాచిత్రం, obj_కొలత);
    నిర్మాణం ggml_cgraph * cgraph = (నిర్మాణం ggml_cgraph *) ((అక్షరం *) ctx->mem_బఫరు + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_కొలత = ggml_hash_కొలత(కొలత * 2);

    శూన్య * p = cgraph + 1;

    నిర్మాణం ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, కొలత      * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *));
    నిర్మాణం ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, కొలత      * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(int32_t), పరిమాణం(int32_t));
    నిర్మాణం ggml_tensor ** hash_కీలకాలు_ptr  =         incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *));
    నిర్మాణం ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)) : ఖాళీ;
    నిర్మాణం ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_కొలత * పరిమాణం(నిర్మాణం ggml_tensor *), పరిమాణం(నిర్మాణం ggml_tensor *)) : ఖాళీ;

    ggml_bitset_t * hash_వాడిన = incr_ptr_aligned(&p, ggml_bitset_కొలత(hash_కొలత) * పరిమాణం(ggml_bitset_t), పరిమాణం(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    ప్రతిజ్ఞా(obj_కొలత == (size_t)((అక్షరం *)p - (అక్షరం *)cgraph));

    *cgraph = (నిర్మాణం ggml_cgraph) {
        /*.size         =*/ కొలత,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_కొలత, hash_వాడిన, hash_కీలకాలు_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_క్రమం_ఎడమ_గమ్యం_కుడి,
        /*.uid          =*/ 0,
    };

    ggml_hash_సమితి_reset(&cgraph->సందర్శించిన_hash_సమితి);
    ఒకవేళ (grads) {
        memset(cgraph->grads,     0, hash_కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
        memset(cgraph->grad_accs, 0, hash_కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
    }

    తిరిగి cgraph;
}

నిర్మాణం ggml_cgraph * ggml_new_లేఖాచిత్రం(నిర్మాణం ggml_context * ctx) {
    తిరిగి ggml_new_లేఖాచిత్రం_custom(ctx, GGML_DEFAULT_లేఖాచిత్రం_కొలత, false);
}

నిర్మాణం ggml_cgraph ggml_లేఖాచిత్రం_view(నిర్మాణం ggml_cgraph * cgraph0, పూర్ణాంకం i0, పూర్ణాంకం i1) {
    నిర్మాణం ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ ఖాళీ, // gradients would need visited_hash_set
        /*.grad_accs        =*/ ఖాళీ,
        /*.leafs            =*/ ఖాళీ,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->సందర్శించిన_hash_సమితి,
        /*.order            =*/ cgraph0->క్రమం,
        /*.uid              =*/ 0
    };

    తిరిగి cgraph;
}

శూన్య ggml_లేఖాచిత్రం_cpy(నిర్మాణం ggml_cgraph * src, నిర్మాణం ggml_cgraph * dst) {
    GGML_ASSERT(dst->కొలత >= src->n_leafs);
    GGML_ASSERT(dst->కొలత >= src->n_nodes);
    GGML_ASSERT(dst->సందర్శించిన_hash_సమితి.కొలత >= src->సందర్శించిన_hash_సమితి.కొలత);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->క్రమం   = src->క్రమం;

    కొరకు (పూర్ణాంకం i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    కొరకు (పూర్ణాంకం i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    కొరకు (size_t i = 0; i < src->సందర్శించిన_hash_సమితి.కొలత; ++i) {
        // copy all hashset keys (tensors) that are in use
        ఒకవేళ (ggml_bitset_get(src->సందర్శించిన_hash_సమితి.వాడిన, i)) {
            size_t new_hash_pos = ggml_hash_చొప్పించు(&dst->సందర్శించిన_hash_సమితి, src->సందర్శించిన_hash_సమితి.కీలకాలు[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    ఒకవేళ (dst->grads) {
        memset(dst->grads,     0, dst->సందర్శించిన_hash_సమితి.కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
        memset(dst->grad_accs, 0, dst->సందర్శించిన_hash_సమితి.కొలత*పరిమాణం(నిర్మాణం ggml_tensor *));
    }
    ఒకవేళ (src->grads) {
        GGML_ASSERT(dst->grads     != ఖాళీ);
        GGML_ASSERT(dst->grad_accs != ఖాళీ);
        కొరకు (పూర్ణాంకం i = 0; i < src->n_nodes; ++i) {
            స్థిరరాశి size_t igrad_src = ggml_hash_కనుగొను(&src->సందర్శించిన_hash_సమితి, src->nodes[i]);
            స్థిరరాశి size_t igrad_dst = ggml_hash_కనుగొను(&dst->సందర్శించిన_hash_సమితి, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_నిండిన);
            GGML_ASSERT(ggml_bitset_get(src->సందర్శించిన_hash_సమితి.వాడిన, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_నిండిన);
            GGML_ASSERT(ggml_bitset_get(dst->సందర్శించిన_hash_సమితి.వాడిన, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

నిర్మాణం ggml_cgraph * ggml_లేఖాచిత్రం_dup(నిర్మాణం ggml_context * ctx, నిర్మాణం ggml_cgraph * cgraph, bool force_grads) {
    నిర్మాణం ggml_cgraph * ఫలితం = ggml_new_లేఖాచిత్రం_custom(ctx, cgraph->కొలత, cgraph->grads || force_grads);
    ggml_లేఖాచిత్రం_cpy(cgraph, ఫలితం);
    తిరిగి ఫలితం;
}

నిర్మాణం ggml_tensor * ggml_సమితి_zero(నిర్మాణం ggml_tensor * tensor) {
    ఒకవేళ (ggml_is_ఖాళీలు(tensor)) {
        తిరిగి tensor;
    }
    ఒకవేళ (tensor->బఫరు) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } లేకపోతే {
        GGML_ASSERT(tensor->దత్తాంశం);
        memset(tensor->దత్తాంశం, 0, ggml_nbytes(tensor));
    }
    తిరిగి tensor;
}

శూన్య ggml_లేఖాచిత్రం_reset(నిర్మాణం ggml_cgraph * cgraph) {
    ఒకవేళ (!cgraph) {
        తిరిగి;
    }
    GGML_ASSERT(cgraph->grads != ఖాళీ);

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు     = cgraph->nodes[i];
        నిర్మాణం ggml_tensor * grad_acc = ggml_లేఖాచిత్రం_get_grad_acc(cgraph, కణుపు);

        ఒకవేళ (కణుపు->op == GGML_OP_OPT_అడుగు_ADAMW) {
            // clear momenta
            ggml_సమితి_zero(కణుపు->src[2]);
            ggml_సమితి_zero(కణుపు->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        ఒకవేళ (grad_acc) {
            ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                స్థిరరాశి దశాంశం onef = 1.0f;
                ఒకవేళ (grad_acc->బఫరు) {
                    ggml_backend_tensor_సమితి(grad_acc, &onef, 0, పరిమాణం(దశాంశం));
                } లేకపోతే {
                    GGML_ASSERT(grad_acc->దత్తాంశం);
                    *((దశాంశం *) grad_acc->దత్తాంశం) = onef;
                }
            } లేకపోతే {
                ggml_సమితి_zero(grad_acc);
            }
        }
    }
}

శూన్య ggml_లేఖాచిత్రం_clear(నిర్మాణం ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_సమితి_reset(&cgraph->సందర్శించిన_hash_సమితి);
}

పూర్ణాంకం ggml_లేఖాచిత్రం_కొలత(నిర్మాణం ggml_cgraph * cgraph) {
    తిరిగి cgraph->కొలత;
}

నిర్మాణం ggml_tensor * ggml_లేఖాచిత్రం_కణుపు(నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం i) {
    ఒకవేళ (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        తిరిగి cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    తిరిగి cgraph->nodes[i];
}

నిర్మాణం ggml_tensor ** ggml_లేఖాచిత్రం_nodes(నిర్మాణం ggml_cgraph * cgraph) {
    తిరిగి cgraph->nodes;
}

పూర్ణాంకం ggml_లేఖాచిత్రం_n_nodes(నిర్మాణం ggml_cgraph * cgraph) {
    తిరిగి cgraph->n_nodes;
}

శూన్య ggml_లేఖాచిత్రం_కూడు_కణుపు(నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->కొలత > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

నిర్మాణం ggml_tensor * ggml_లేఖాచిత్రం_get_tensor(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి అక్షరం * name) {
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_leafs; i++) {
        నిర్మాణం ggml_tensor * ఆకు = cgraph->leafs[i];

        ఒకవేళ (strcmp(ఆకు->name, name) == 0) {
            తిరిగి ఆకు;
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        ఒకవేళ (strcmp(కణుపు->name, name) == 0) {
            తిరిగి కణుపు;
        }
    }

    తిరిగి ఖాళీ;
}

నిర్మాణం ggml_tensor * ggml_లేఖాచిత్రం_get_grad(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు) {
    స్థిరరాశి size_t igrad = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు);
    తిరిగి igrad != GGML_HASHSET_నిండిన && ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, igrad) && cgraph->grads ? cgraph->grads[igrad] : ఖాళీ;
}

నిర్మాణం ggml_tensor * ggml_లేఖాచిత్రం_get_grad_acc(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు) {
    స్థిరరాశి size_t igrad = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు);
    తిరిగి igrad != GGML_HASHSET_నిండిన && ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : ఖాళీ;
}

శూన్య ggml_లేఖాచిత్రం_print(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                కణుపు->ne[0], కణుపు->ne[1], కణుపు->ne[2],
                ggml_op_name(కణుపు->op), (కణుపు->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_లేఖాచిత్రం_get_grad(cgraph, కణుపు) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_leafs; i++) {
        నిర్మాణం ggml_tensor * కణుపు = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                కణుపు->ne[0], కణుపు->ne[1],
                ggml_op_name(కణుపు->op),
                ggml_get_name(కణుపు));
    }

    GGML_LOG_INFO("========================================\n");
}

స్థిర పూర్ణాంకం ggml_కణుపు_list_కనుగొను_tensor(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                      స్థిరరాశి పూర్ణాంకం *                idxs,
                                      పూర్ణాంకం                        లెక్క,
                                      స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    కొరకు (పూర్ణాంకం i = 0; i < లెక్క; ++i) {
        స్థిరరాశి పూర్ణాంకం కణుపు_idx = idxs[i];

        ఒకవేళ (కణుపు_idx >= cgraph->n_nodes) {
            తిరిగి -1;
        }
        ఒకవేళ (cgraph->nodes[కణుపు_idx] == tensor) {
            తిరిగి i;
        }
    }
    తిరిగి -1;
}

స్థిర bool ggml_is_constant(స్థిరరాశి నిర్మాణం ggml_tensor * tensor) {
    తిరిగి tensor->బఫరు != ఖాళీ && ggml_backend_బఫరు_get_usage(tensor->బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                స్థిరరాశి పూర్ణాంకం *                కణుపు_idxs,
                                పూర్ణాంకం                        లెక్క,
                                స్థిరరాశి గణన ggml_op *       ops,
                                స్థిరరాశి పూర్ణాంకం *                outputs,
                                పూర్ణాంకం                        అంకె_outputs) {
    GGML_ASSERT(outputs && అంకె_outputs > 0);

    కొరకు (పూర్ణాంకం i = 0; i < లెక్క; ++i) {
        ఒకవేళ (కణుపు_idxs[i] >= cgraph->n_nodes) {
            తిరిగి false;
        }

        స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idxs[i]];

        ఒకవేళ (కణుపు->op != ops[i]) {
            తిరిగి false;
        }

        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            తిరిగి false;
        }

        ఒకవేళ (ggml_కణుపు_list_కనుగొను_tensor(cgraph, outputs, అంకె_outputs, కణుపు) != -1) {
            కొనసాగు;
        }

        ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_OUTPUT) {
            తిరిగి false;
        }

        పూర్ణాంకం subgraph_ఉపయోగిస్తుంది = 0;
        కొరకు (పూర్ణాంకం j = i + 1; j < లెక్క; ++j) {
            స్థిరరాశి నిర్మాణం ggml_tensor * other_కణుపు = cgraph->nodes[కణుపు_idxs[j]];
            కొరకు (పూర్ణాంకం src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                ఒకవేళ (other_కణుపు->src[src_idx] == కణుపు) {
                    subgraph_ఉపయోగిస్తుంది++;
                }
            }
        }

        ఒకవేళ (subgraph_ఉపయోగిస్తుంది != ggml_కణుపు_get_use_లెక్క(cgraph, కణుపు_idxs[i])) {
            తిరిగి false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        నిర్మాణం ggml_tensor * view_src = కణుపు->view_src;
        అయితే (view_src) {
            ఒకవేళ (ggml_కణుపు_list_కనుగొను_tensor(cgraph, కణుపు_idxs, లెక్క, view_src) == -1 && !ggml_is_constant(view_src)) {
                తిరిగి false;
            }
            view_src = view_src->view_src;
        }
    }

    తిరిగి true;
}

// check if node is part of the graph
స్థిర bool ggml_లేఖాచిత్రం_కనుగొను(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు) {
    ఒకవేళ (cgraph == ఖాళీ) {
        తిరిగి true;
    }

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ఒకవేళ (cgraph->nodes[i] == కణుపు) {
            తిరిగి true;
        }
    }

    తిరిగి false;
}

స్థిర నిర్మాణం ggml_tensor * ggml_లేఖాచిత్రం_get_మాతృ(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు) {
    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        నిర్మాణం ggml_tensor * మాతృ = cgraph->nodes[i];
        నిర్మాణం ggml_tensor * grad = ggml_లేఖాచిత్రం_get_grad(cgraph, మాతృ);

        ఒకవేళ (grad == కణుపు) {
            తిరిగి మాతృ;
        }
    }

    తిరిగి ఖాళీ;
}

స్థిర శూన్య ggml_లేఖాచిత్రం_dump_dot_కణుపు_అంచు(FILE * fp, స్థిరరాశి నిర్మాణం ggml_cgraph * gb, నిర్మాణం ggml_tensor * కణుపు, నిర్మాణం ggml_tensor * మాతృ, స్థిరరాశి అక్షరం * label)  {
    నిర్మాణం ggml_tensor * gparent = ggml_లేఖాచిత్రం_get_మాతృ(gb, కణుపు);
    నిర్మాణం ggml_tensor * gparent0 = ggml_లేఖాచిత్రం_get_మాతృ(gb, మాతృ);
    fprintf(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (శూన్య *) gparent0 : (శూన్య *) మాతృ,
            gparent ? (శూన్య *) gparent : (శూన్య *) కణుపు,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

స్థిర శూన్య ggml_లేఖాచిత్రం_dump_dot_ఆకు_అంచు(FILE * fp, నిర్మాణం ggml_tensor * కణుపు, నిర్మాణం ggml_tensor * మాతృ, స్థిరరాశి అక్షరం * label)  {
    fprintf(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (శూన్య *) మాతృ,
            (శూన్య *) కణుపు,
            label);
}

శూన్య ggml_లేఖాచిత్రం_dump_dot(స్థిరరాశి నిర్మాణం ggml_cgraph * gb, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి అక్షరం * filename) {
    అక్షరం color[16];

    FILE * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    కొరకు (పూర్ణాంకం i = 0; i < gb->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = gb->nodes[i];
        నిర్మాణం ggml_tensor * grad = ggml_లేఖాచిత్రం_get_grad(gb, కణుపు);

        ఒకవేళ (ggml_లేఖాచిత్రం_get_మాతృ(gb, కణుపు) != ఖాళీ) {
            కొనసాగు;
        }

        ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, పరిమాణం(color), "yellow");
        } లేకపోతే ఒకవేళ (grad) {
            ఒకవేళ (ggml_లేఖాచిత్రం_కనుగొను(cgraph, కణుపు)) {
                snprintf(color, పరిమాణం(color), "green");
            } లేకపోతే {
                snprintf(color, పరిమాణం(color), "lightblue");
            }
        } లేకపోతే {
            snprintf(color, పరిమాణం(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (శూన్య *) కణుపు, color);

        ఒకవేళ (అక్షరశ్రేణి_పొడవు(కణుపు->name) > 0) {
            fprintf(fp, "%s (%s)|", కణుపు->name, ggml_type_name(కణుపు->type));
        } లేకపోతే {
            fprintf(fp, "(%s)|", ggml_type_name(కణుపు->type));
        }

        ఒకవేళ (ggml_is_మాతృక(కణుపు)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, కణుపు->ne[0], కణుపు->ne[1], ggml_op_చిహ్నం(కణుపు->op));
        } లేకపోతే {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, కణుపు->ne[0], కణుపు->ne[1], కణుపు->ne[2], ggml_op_చిహ్నం(కణుపు->op));
        }

        ఒకవేళ (grad) {
            fprintf(fp, " | <g>%s\"; ]\n", ggml_op_చిహ్నం(grad->op));
        } లేకపోతే {
            fprintf(fp, "\"; ]\n");
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < gb->n_leafs; i++) {
        నిర్మాణం ggml_tensor * కణుపు = gb->leafs[i];

        snprintf(color, పరిమాణం(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (శూన్య *) కణుపు, color);

        ఒకవేళ (అక్షరశ్రేణి_పొడవు(కణుపు->name) > 0) {
            fprintf(fp, "%s (%s)|", కణుపు->name, ggml_type_name(కణుపు->type));
        } లేకపోతే {
            fprintf(fp, "(%s)|", ggml_type_name(కణుపు->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, కణుపు->ne[0], కణుపు->ne[1]);
        ఒకవేళ (ggml_nelements(కణుపు) < 5 && కణుపు->దత్తాంశం != ఖాళీ) {
            fprintf(fp, " | (");
            కొరకు (పూర్ణాంకం j = 0; j < ggml_nelements(కణుపు); j++) {
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
                ఒకవేళ (j < ggml_nelements(కణుపు) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    కొరకు (పూర్ణాంకం i = 0; i < gb->n_nodes; i++) {
        నిర్మాణం ggml_tensor * కణుపు = gb->nodes[i];

        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ఒకవేళ (కణుపు->src[j]) {
                అక్షరం label[16];
                snprintf(label, పరిమాణం(label), "src %d", j);
                ggml_లేఖాచిత్రం_dump_dot_కణుపు_అంచు(fp, gb, కణుపు, కణుపు->src[j], label);
            }
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < gb->n_leafs; i++) {
        నిర్మాణం ggml_tensor * కణుపు = gb->leafs[i];

        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ఒకవేళ (కణుపు->src[j]) {
                అక్షరం label[16];
                snprintf(label, పరిమాణం(label), "src %d", j);
                ggml_లేఖాచిత్రం_dump_dot_ఆకు_అంచు(fp, కణుపు, కణుపు->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

శూన్య ggml_సమితి_ఆదానం(నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_ఆదానం;
}

శూన్య ggml_సమితి_output(నిర్మాణం ggml_tensor * tensor) {
    కొరకు (నిర్మాణం ggml_tensor * cur = tensor; cur != ఖాళీ; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

శూన్య ggml_సమితి_param(నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

శూన్య ggml_సమితి_loss(నిర్మాణం ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

శూన్య ggml_quantize_init(గణన ggml_type type) {
    ggml_critical_section_ప్రారంభం();

    ఎంపిక (type) {
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); ఆపు;
        స్థితి GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); ఆపు;
        అప్రమేయ: // nothing
            ఆపు;
    }

    ggml_critical_section_చివర();
}

శూన్య ggml_quantize_free(శూన్య) {
    ggml_critical_section_ప్రారంభం();

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);

    ggml_critical_section_చివర();
}

bool ggml_quantize_requires_imatrix(గణన ggml_type type) {
    తిరిగి
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

size_t ggml_quantize_chunk(
        గణన ggml_type   type,
           స్థిరరాశి దశాంశం * src,
                  శూన్య * dst,
               int64_t   ప్రారంభం,
               int64_t   nrows,
               int64_t   n_చొప్పున_అడ్డువరుస,
           స్థిరరాశి దశాంశం * imatrix) {
    స్థిరరాశి int64_t n = nrows * n_చొప్పున_అడ్డువరుస;

    ఒకవేళ (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != ఖాళీ);
    }

    GGML_ASSERT(ప్రారంభం % type_traits[type].blck_కొలత == 0);
    GGML_ASSERT(ప్రారంభం % n_చొప్పున_అడ్డువరుస == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    స్థిరరాశి size_t ప్రారంభం_అడ్డువరుస = ప్రారంభం / n_చొప్పున_అడ్డువరుస;
    స్థిరరాశి size_t అడ్డువరుస_కొలత  = ggml_అడ్డువరుస_కొలత(type, n_చొప్పున_అడ్డువరుస);

    size_t ఫలితం = 0;

    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:    ఫలితం = quantize_q1_0   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q2_0:    ఫలితం = quantize_q2_0   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q4_0:    ఫలితం = quantize_q4_0   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q4_1:    ఫలితం = quantize_q4_1   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q5_0:    ఫలితం = quantize_q5_0   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q5_1:    ఫలితం = quantize_q5_1   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q8_0:    ఫలితం = quantize_q8_0   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_MXFP4:   ఫలితం = quantize_mxfp4  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_NVFP4:   ఫలితం = quantize_nvfp4  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q2_K:    ఫలితం = quantize_q2_K   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q3_K:    ఫలితం = quantize_q3_K   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q4_K:    ఫలితం = quantize_q4_K   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q5_K:    ఫలితం = quantize_q5_K   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_Q6_K:    ఫలితం = quantize_q6_K   (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_TQ1_0:   ఫలితం = quantize_tq1_0  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_TQ2_0:   ఫలితం = quantize_tq2_0  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ2_XXS: ఫలితం = quantize_iq2_xxs(src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ2_XS:  ఫలితం = quantize_iq2_xs (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS: ఫలితం = quantize_iq3_xxs(src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ3_S:   ఫలితం = quantize_iq3_s  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ2_S:   ఫలితం = quantize_iq2_s  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ1_S:   ఫలితం = quantize_iq1_s  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ1_M:   ఫలితం = quantize_iq1_m  (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ4_NL:  ఫలితం = quantize_iq4_nl (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_IQ4_XS:  ఫలితం = quantize_iq4_xs (src + ప్రారంభం, (అక్షరం *) dst + ప్రారంభం_అడ్డువరుస * అడ్డువరుస_కొలత, nrows, n_చొప్పున_అడ్డువరుస, imatrix); ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                size_t elemsize = పరిమాణం(ggml_fp16_t);
                ggml_fp32_గమ్యం_fp16_అడ్డువరుస(src + ప్రారంభం, (ggml_fp16_t *)dst + ప్రారంభం, n);
                ఫలితం = n * elemsize;
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                size_t elemsize = పరిమాణం(ggml_bf16_t);
                ggml_fp32_గమ్యం_bf16_అడ్డువరుస_ref(src + ప్రారంభం, (ggml_bf16_t *)dst + ప్రారంభం, n);
                ఫలితం = n * elemsize;
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                size_t elemsize = పరిమాణం(దశాంశం);
                ఫలితం = n * elemsize;
                జ్ఞాపకం_నకలు((uint8_t *)dst + ప్రారంభం * elemsize, src + ప్రారంభం, ఫలితం);
            } ఆపు;
        అప్రమేయ:
            ప్రతిజ్ఞా(false);
    }

    GGML_ASSERT(ఫలితం == nrows * అడ్డువరుస_కొలత);

    తిరిగి ఫలితం;
}

////////////////////////////////////////////////////////////////////////////////

శూన్య ggml_log_get(ggml_log_callback * log_callback, శూన్య ** user_దత్తాంశం) {
    *log_callback = g_logger_state.log_callback;
    *user_దత్తాంశం    = g_logger_state.log_callback_user_దత్తాంశం;
}

శూన్య ggml_log_సమితి(ggml_log_callback log_callback, శూన్య * user_దత్తాంశం) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_దత్తాంశం = user_దత్తాంశం;
}

శూన్య ggml_threadpool_params_init(నిర్మాణం ggml_threadpool_params * p, పూర్ణాంకం n_దారాలు) {
    p->n_దారాలు  = n_దారాలు;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    memset(p->cpumask, 0, GGML_MAX_N_దారాలు); // all-zero means use the default affinity (usually inherited)
}

నిర్మాణం ggml_threadpool_params ggml_threadpool_params_default(పూర్ణాంకం n_దారాలు) {
    నిర్మాణం ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_దారాలు);
    తిరిగి p;
}

bool ggml_threadpool_params_సరిపోలిక(స్థిరరాశి నిర్మాణం ggml_threadpool_params * p0, స్థిరరాశి నిర్మాణం ggml_threadpool_params * p1) {
    ఒకవేళ (p0->n_దారాలు  != p1->n_దారాలు  ) తిరిగి false;
    ఒకవేళ (p0->prio       != p1->prio       ) తిరిగి false;
    ఒకవేళ (p0->poll       != p1->poll       ) తిరిగి false;
    ఒకవేళ (p0->strict_cpu != p1->strict_cpu ) తిరిగి false;
    తిరిగి memcmp(p0->cpumask, p1->cpumask, GGML_MAX_N_దారాలు) == 0;
}
