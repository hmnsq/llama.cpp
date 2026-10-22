#घोषणा _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#घोषणा _USE_MATH_DEFINES // For M_PI on MSVC

#समावेश "ggml-version.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-threading.h"
#समावेश "ggml-cpu.h"
#समावेश "ggml.h"

// FIXME: required here for quantization functions
#समावेश "ggml-quants.h"

#अगर_घोषित GGML_USE_CPU_HBM
#समावेश <hbwmalloc.h>
#पूर्ण_अगर

#अगर defined(_MSC_VER) || defined(__MINGW32__)
#समावेश <malloc.h> // using malloc.h with MSC/MINGW
#या_अगर !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#समावेश <alloca.h>
#पूर्ण_अगर

#समावेश <assert.h>
#समावेश <errno.h>
#समावेश <time.h>
#समावेश <math.h>
#समावेश <stdlib.h>
#समावेश <string.h>
#समावेश <stdint.h>
#समावेश <inttypes.h>
#समावेश <stdio.h>
#समावेश <float.h>
#समावेश <limits.h>
#समावेश <stdarg.h>
#समावेश <signal.h>
#अगर defined(__gnu_linux__)
#समावेश <syscall.h>
#पूर्ण_अगर

#अगर defined(__APPLE__)
#समावेश <unistd.h>
#समावेश <mach/mach.h>
#समावेश <TargetConditionals.h>
#पूर्ण_अगर

#अगर defined(_WIN32)
#घोषणा WIN32_LEAN_AND_MEAN
#अगर_अघोषित NOMINMAX
    #घोषणा NOMINMAX
#पूर्ण_अगर
#समावेश <windows.h>
#पूर्ण_अगर

#घोषणा UNUSED GGML_UNUSED

uint64_t ggml_graph_अग्र_uid(व्योम) {
#अगर_घोषित _MSC_VER
#अगर defined(_WIN32)
    अटल अस्थिर LONG गणक = 1;
    वापस (uint64_t) InterlockedIncrement(&गणक) - 1;
#अन्यथा
    अटल अस्थिर दीर्घ दीर्घ गणक = 1;
    वापस (uint64_t) _InterlockedIncrement64(&गणक) - 1;
#पूर्ण_अगर
#अन्यथा
    अटल uint64_t गणक = 1;
    वापस __atomic_fetch_add(&गणक, 1, __ATOMIC_RELAXED);
#पूर्ण_अगर
}

// Needed for ggml_fp32_to_bf16_row()
#अगर defined(__AVX512BF16__)
#अगर defined(_MSC_VER)
#घोषणा m512i(p) p
#अन्यथा
#समावेश <immintrin.h>
#घोषणा m512i(p) (__m512i)(p)
#पूर्ण_अगर // defined(_MSC_VER)
#पूर्ण_अगर // defined(__AVX512BF16__)

#अगर defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#समावेश <unistd.h>
#समावेश <sys/types.h>
#समावेश <sys/stat.h>
#समावेश <sys/wait.h>
#अगर defined(__linux__)
#समावेश <sys/prctl.h>
#पूर्ण_अगर

#अगर defined(__ANDROID__)
#समावेश <unwind.h>
#समावेश <dlfcn.h>
#समावेश <stdio.h>

काष्ठा backtrace_state {
    व्योम ** current;
    व्योम ** end;
};

अटल _Unwind_Reason_Code unwind_callback(काष्ठा _Unwind_Context* context, व्योम* arg) {
    काष्ठा backtrace_state * state = (काष्ठा backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    अगर (pc) {
        अगर (state->current == state->end) {
            वापस URC_END_OF_ढेर;
        } अन्यथा {
            *state->current++ = (व्योम*)pc;
        }
    }
    वापस _URC_NO_REASON;
}

अटल व्योम ggml_print_backtrace_symbols(व्योम) {
    स्थिर पूर्णांक max = 100;
    व्योम* बफर[max];

    काष्ठा backtrace_state state = {बफर, बफर + max};
    _Unwind_Backtrace(unwind_callback, &state);

    पूर्णांक गिनती = state.current - बफर;

    क्रम (पूर्णांक idx = 0; idx < गिनती; ++idx) {
        स्थिर व्योम * addr = बफर[idx];
        स्थिर अक्षर * प्रतीक = "";

        Dl_info info;
        अगर (dladdr(addr, &info) && info.dli_sname) {
            प्रतीक = info.dli_sname;
        }

        ख_लिखो(मानक_त्रुटि, "%d: %p %s\n", idx, addr, प्रतीक);
    }
}
#या_अगर defined(__linux__) && defined(__GLIBC__)
#समावेश <execinfo.h>
अटल व्योम ggml_print_backtrace_symbols(व्योम) {
    व्योम * trace[100];
    पूर्णांक nptrs = backtrace(trace, माप(trace)/माप(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#या_अगर defined(__APPLE__)
#समावेश <execinfo.h>
अटल व्योम ggml_print_backtrace_symbols(व्योम) {
    व्योम * trace[100];
    पूर्णांक nptrs = backtrace(trace, माप(trace)/माप(trace[0]));
    backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#अन्यथा
अटल व्योम ggml_print_backtrace_symbols(व्योम) {
    // platform not supported
}
#पूर्ण_अगर

व्योम ggml_print_backtrace(व्योम) {
    स्थिर अक्षर * GGML_NO_BACKTRACE = दो_पर्या("GGML_NO_BACKTRACE");
    अगर (GGML_NO_BACKTRACE) {
        वापस;
    }
#अगर defined(__APPLE__)
    // On macOS, fork+debugger attachment is problematic due to:
    // 1. libdispatch "poisons" forked child processes
    // 2. lldb has issues attaching to parent from forked child
    // Use simple backtrace() instead to avoid Terminal.app crashes
    स्थिर अक्षर * GGML_BACKTRACE_LLDB = दो_पर्या("GGML_BACKTRACE_LLDB");
    अगर (!GGML_BACKTRACE_LLDB) {
        ख_लिखो(मानक_त्रुटि, "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.\n");
        ख_लिखो(मानक_त्रुटि, "WARNING: GGML_BACKTRACE_LLDB may cause native MacOS Terminal.app to crash.\n");
        ख_लिखो(मानक_त्रुटि, "See: https://github.com/ggml-org/llama.cpp/pull/17869\n");
        ggml_print_backtrace_symbols();
        वापस;
    }
#पूर्ण_अगर
#अगर defined(__linux__)
    खाता * f = ख_खोलो("/proc/self/status", "r");
    माप_प्रकार आकार = 0;
    अक्षर * पंक्ति = शून्य;
    ssize_t लंबाई = 0;
    जबतक ((लंबाई = getline(&पंक्ति, &आकार, f)) > 0) {
        अगर (!म_भेदन(पंक्ति, "TracerPid:", माप("TracerPid:") - 1) &&
            (लंबाई != माप("TracerPid:\t0\n") - 1 || पंक्ति[लंबाई - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            मुक्त(पंक्ति);
            ख_बंद(f);
            वापस;
        }
    }
    मुक्त(पंक्ति);
    ख_बंद(f);
    पूर्णांक lock[2] = { -1, -1 };
    (व्योम) !pipe(lock); // Don't start gdb until after PR_SET_PTRACER
#पूर्ण_अगर
    स्थिर पूर्णांक parent_pid = getpid();
    स्थिर पूर्णांक child_pid = विभाजन();
    अगर (child_pid < 0) { // error
#अगर defined(__linux__)
        बंद(lock[1]);
        बंद(lock[0]);
#पूर्ण_अगर
        वापस;
    } अन्यथा अगर (child_pid == 0) { // child
        अक्षर attach[32];
        snprintf(attach, माप(attach), "attach %d", parent_pid);
#अगर defined(__linux__)
        बंद(lock[1]);
        (व्योम) !पढ़ो(lock[0], lock, 1);
        बंद(lock[0]);
#पूर्ण_अगर
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (अक्षर *) शून्य);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach[माप("attach ") - 1],
            (अक्षर *) शून्य);
        // gdb failed, fallback to backtrace_symbols
        ggml_print_backtrace_symbols();
        _Exit(0);
    } अन्यथा { // parent
#अगर defined(__linux__)
        prctl(PR_SET_PTRACER, child_pid);
        बंद(lock[1]);
        बंद(lock[0]);
#पूर्ण_अगर
        waitpid(child_pid, शून्य, 0);
    }
}
#अन्यथा
व्योम ggml_print_backtrace(व्योम) {
    // platform not supported
}
#पूर्ण_अगर

अटल ggml_abort_callback_t g_abort_callback = शून्य;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t ret_val = g_abort_callback;
    g_abort_callback = callback;
    वापस ret_val;
}

व्योम ggml_abort(स्थिर अक्षर * file, पूर्णांक पंक्ति, स्थिर अक्षर * fmt, ...) {
    ख_साफ(मानक_निकास);

    अक्षर message[2048];
    पूर्णांक offset = snprintf(message, माप(message), "%s:%d: ", file, पंक्ति);

    बहु_सूची args;
    बहु_शुरू(args, fmt);
    vsnprintf(message + offset, माप(message) - offset, fmt, args);
    बहु_पूर्ण(args);

    अगर (g_abort_callback) {
        g_abort_callback(message);
    } अन्यथा {
        // default: print error and backtrace to stderr
        ख_लिखो(मानक_त्रुटि, "%s\n", message);
        ggml_print_backtrace();
    }

    पात();
}

// ggml_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//

काष्ठा ggml_logger_state {
    ggml_log_callback log_callback;
    व्योम * log_callback_user_तथ्य;
};
अटल काष्ठा ggml_logger_state g_logger_state = {ggml_log_callback_default, शून्य};

अटल व्योम ggml_log_internal_v(क्रमागत ggml_log_level level, स्थिर अक्षर * format, बहु_सूची args) {
    अगर (format == शून्य) {
        वापस;
    }
    बहु_सूची args_copy;
    va_copy(args_copy, args);
    अक्षर बफर[128];
    पूर्णांक len = vsnprintf(बफर, 128, format, args);
    अगर (len < 128) {
        g_logger_state.log_callback(level, बफर, g_logger_state.log_callback_user_तथ्य);
    } अन्यथा {
        अक्षर * buffer2 = (अक्षर *) सुस्मृति(len + 1, माप(अक्षर));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_तथ्य);
        मुक्त(buffer2);
    }
    बहु_पूर्ण(args_copy);
}

व्योम ggml_log_internal(क्रमागत ggml_log_level level, स्थिर अक्षर * format, ...) {
    बहु_सूची args;
    बहु_शुरू(args, format);
    ggml_log_internal_v(level, format, args);
    बहु_पूर्ण(args);
}

व्योम ggml_log_callback_default(क्रमागत ggml_log_level level, स्थिर अक्षर * text, व्योम * user_तथ्य) {
    (व्योम) level;
    (व्योम) user_तथ्य;
    ख_माला_दो(text, मानक_त्रुटि);
    ख_साफ(मानक_त्रुटि);
}

//
// end of logging block
//

#अगर_घोषित GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#पूर्ण_अगर


// the thread sanitizer does not intercept vm_allocate and vm_deallocate, so it produces false positives
#अगर defined(__has_feature)
#अगर __has_feature(thread_sanitizer)
#घोषणा GGML_TSAN_ENABLED 1
#पूर्ण_अगर
#अन्यथा
#अगर defined(__SANITIZE_THREAD__)
#घोषणा GGML_TSAN_ENABLED 1
#पूर्ण_अगर
#पूर्ण_अगर

व्योम * ggml_aligned_malloc(माप_प्रकार आकार) {
#अगर defined(__s390x__)
    स्थिर पूर्णांक alignment = 256;
#अन्यथा
    स्थिर पूर्णांक alignment = 64;
#पूर्ण_अगर

#अगर defined(_MSC_VER) || defined(__MINGW32__)
    वापस _aligned_malloc(आकार, alignment);
#अन्यथा
    अगर (आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        वापस शून्य;
    }
    व्योम * aligned_memory = शून्य;
  #अगर_घोषित GGML_USE_CPU_HBM
    पूर्णांक परिणाम = hbw_posix_memalign(&aligned_memory, alignment, आकार);
  #या_अगर TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, आकार, VM_FLAGS_ANYWHERE);
    पूर्णांक परिणाम = EFAULT;
    चयन (alloc_status) {
        हाल KERN_SUCCESS:
            परिणाम = 0;
            अवरोध;
        हाल KERN_INVALID_ADDRESS:
            परिणाम = EINVAL;
            अवरोध;
        हाल KERN_NO_SPACE:
            परिणाम = ENOMEM;
            अवरोध;
        शेष:
            परिणाम = EFAULT;
            अवरोध;
    }
  #अन्यथा
    पूर्णांक परिणाम = posix_memalign(&aligned_memory, alignment, आकार);
  #पूर्ण_अगर
    अगर (परिणाम != 0) {
        // Handle allocation failure
        स्थिर अक्षर *error_desc = "unknown allocation error";
        चयन (परिणाम) {
            हाल EINVAL:
                error_desc = "invalid alignment value";
                अवरोध;
            हाल ENOMEM:
                error_desc = "insufficient memory";
                अवरोध;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, आकार/(1024.0*1024.0));
        वापस शून्य;
    }
    वापस aligned_memory;
#पूर्ण_अगर
}

व्योम ggml_aligned_free(व्योम * ptr, माप_प्रकार आकार) {
    GGML_UNUSED(आकार);
#अगर defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#या_अगर GGML_USE_CPU_HBM
    अगर (ptr != शून्य) {
        hbw_free(ptr);
    }
#या_अगर TARGET_OS_OSX && !defined(GGML_TSAN_ENABLED)
    अगर (ptr != शून्य) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, आकार);
    }
#अन्यथा
    मुक्त(ptr);
#पूर्ण_अगर
}


अंतरभूत अटल व्योम * ggml_malloc(माप_प्रकार आकार) {
    अगर (आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_malloc!\n");
        वापस शून्य;
    }
    व्योम * परिणाम = दो_स्मृति(आकार);
    अगर (परिणाम == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, आकार/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    वापस परिणाम;
}

// calloc
अंतरभूत अटल व्योम * ggml_calloc(माप_प्रकार अंक, माप_प्रकार आकार) {
    अगर (अंक == 0 || आकार == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_calloc!\n");
        वापस शून्य;
    }
    व्योम * परिणाम = सुस्मृति(अंक, आकार);
    अगर (परिणाम == शून्य) {
        GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, आकार/(1024.0*1024.0));
        GGML_ABORT("fatal error");
    }
    वापस परिणाम;
}

#घोषणा GGML_MALLOC(आकार)      ggml_malloc(आकार)
#घोषणा GGML_CALLOC(अंक, आकार) ggml_calloc(अंक, आकार)

#घोषणा GGML_FREE(ptr) मुक्त(ptr)

स्थिर अक्षर * ggml_status_to_string(क्रमागत ggml_status status) {
    चयन (status) {
        हाल GGML_STATUS_ALLOC_FAILED: वापस "GGML status: error (failed to allocate memory)";
        हाल GGML_STATUS_FAILED:       वापस "GGML status: error (operation failed)";
        हाल GGML_STATUS_SUCCESS:      वापस "GGML status: success";
        हाल GGML_STATUS_ABORTED:      वापस "GGML status: warning (operation aborted)";
    }

    वापस "GGML status: unknown";
}

भग्न ggml_fp16_to_fp32(ggml_fp16_t x) {
#घोषणा ggml_fp16_to_fp32 do_not_use__ggml_fp16_to_fp32__in_ggml
    वापस GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(भग्न x) {
#घोषणा ggml_fp32_to_fp16 do_not_use__ggml_fp32_to_fp16__in_ggml
    वापस GGML_FP32_TO_FP16(x);
}

भग्न ggml_bf16_to_fp32(ggml_bf16_t x) {
#घोषणा ggml_bf16_to_fp32 do_not_use__ggml_bf16_to_fp32__in_ggml
    वापस GGML_BF16_TO_FP32(x);  // it just left shifts
}

ggml_bf16_t ggml_fp32_to_bf16(भग्न x) {
#घोषणा ggml_fp32_to_bf16 do_not_use__ggml_fp32_to_bf16__in_ggml
    वापस GGML_FP32_TO_BF16(x);
}

व्योम ggml_fp16_to_fp32_row(स्थिर ggml_fp16_t * x, भग्न * y, int64_t n) {
    क्रम (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

व्योम ggml_fp32_to_fp16_row(स्थिर भग्न * x, ggml_fp16_t * y, int64_t n) {
    पूर्णांक i = 0;
    क्रम (; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

व्योम ggml_bf16_to_fp32_row(स्थिर ggml_bf16_t * x, भग्न * y, int64_t n) {
    पूर्णांक i = 0;
    क्रम (; i < n; ++i) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

व्योम ggml_fp32_to_bf16_row_ref(स्थिर भग्न * x, ggml_bf16_t * y, int64_t n) {
    क्रम (पूर्णांक i = 0; i < n; i++) {
        y[i] = ggml_compute_fp32_to_bf16(x[i]);
    }
}

व्योम ggml_fp32_to_bf16_row(स्थिर भग्न * x, ggml_bf16_t * y, int64_t n) {
  पूर्णांक i = 0;
#अगर defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  क्रम (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#पूर्ण_अगर
    क्रम (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b) {
    वापस स_भेद(guid_a, guid_b, माप(ggml_guid)) == 0;
}

स्थिर अक्षर * ggml_version(व्योम) {
    वापस GGML_VERSION;
}

स्थिर अक्षर * ggml_commit(व्योम) {
    वापस GGML_COMMIT;
}

//
// timing
//

#अगर defined(_MSC_VER) || defined(__MINGW32__)
अटल int64_t timer_freq, timer_start;
अटल BOOL CALLBACK ggml_time_init_once(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    UNUSED(once);
    UNUSED(param);
    UNUSED(ctx);

    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    Query_Performance_गणक(&t);
    timer_start = t.QuadPart;

    वापस TRUE;
}
व्योम ggml_time_init(व्योम) {
    अटल INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, ggml_time_init_once, शून्य, शून्य);
}
int64_t ggml_time_ms(व्योम) {
    LARGE_INTEGER t;
    Query_Performance_गणक(&t);
    वापस ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(व्योम) {
    LARGE_INTEGER t;
    Query_Performance_गणक(&t);
    वापस ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#अन्यथा
व्योम ggml_time_init(व्योम) {}
int64_t ggml_time_ms(व्योम) {
    काष्ठा timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    वापस (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(व्योम) {
    काष्ठा timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    वापस (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#पूर्ण_अगर

int64_t ggml_cycles(व्योम) {
    वापस घड़ी();
}

int64_t ggml_cycles_per_ms(व्योम) {
    वापस CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#अगर_घोषित _WIN32
अटल ब_अक्षर_प्रकार * ggml_mbstowcs(स्थिर अक्षर * mbs) {
    पूर्णांक wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, शून्य, 0);
    अगर (!wlen) {
        त्रुटि_सं = EINVAL;
        वापस शून्य;
    }

    ब_अक्षर_प्रकार * wbuf = GGML_MALLOC(wlen * माप(ब_अक्षर_प्रकार));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    अगर (!wlen) {
        GGML_FREE(wbuf);
        त्रुटि_सं = EINVAL;
        वापस शून्य;
    }

    वापस wbuf;
}
#पूर्ण_अगर

खाता * ggml_fopen(स्थिर अक्षर * fname, स्थिर अक्षर * mode) {
#अगर_घोषित _WIN32
    खाता * file = शून्य;

    // convert fname (UTF-8)
    ब_अक्षर_प्रकार * wfname = ggml_mbstowcs(fname);
    अगर (wfname) {
        // convert mode (UTF-8)
        ब_अक्षर_प्रकार * wmode = ggml_mbstowcs(mode);
        अगर (wmode) {
            // open file
            file = _wfopen(wfname, wmode);
            GGML_FREE(wmode);
        }

        GGML_FREE(wfname);
    }

    वापस file;
#अन्यथा
    वापस ख_खोलो(fname, mode);
#पूर्ण_अगर

}

अटल स्थिर काष्ठा ggml_type_traits type_traits[GGML_TYPE_गिनती] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_आकार                = 1,
        .type_आकार                = माप(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_आकार                = 1,
        .type_आकार                = माप(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_आकार                = 1,
        .type_आकार                = माप(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_आकार                = 1,
        .type_आकार                = माप(int64_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_आकार                = 1,
        .type_आकार                = माप(द्विगुन),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_आकार                = 1,
        .type_आकार                = माप(भग्न),
        .is_quantized             = false,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_आकार                = 1,
        .type_आकार                = माप(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
    },
    [GGML_TYPE_Q1_0] = {
        .type_name                = "q1_0",
        .blck_आकार                = QK1_0,
        .type_आकार                = माप(block_q1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q1_0_ref,
    },
    [GGML_TYPE_Q2_0] = {
        .type_name                = "q2_0",
        .blck_आकार                = QK2_0,
        .type_आकार                = माप(block_q2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_0_ref,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_आकार                = QK4_0,
        .type_आकार                = माप(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_आकार                = QK4_1,
        .type_आकार                = माप(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
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
        .type_आकार                = माप(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_आकार                = QK5_1,
        .type_आकार                = माप(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_आकार                = QK8_0,
        .type_आकार                = माप(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_आकार                = QK8_1,
        .type_आकार                = माप(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [GGML_TYPE_MXFP4] = {
        .type_name                = "mxfp4",
        .blck_आकार                = QK_MXFP4,
        .type_आकार                = माप(block_mxfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_mxfp4_ref,
    },
    [GGML_TYPE_NVFP4] = {
        .type_name                = "nvfp4",
        .blck_आकार                = QK_NVFP4,
        .type_आकार                = माप(block_nvfp4),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_nvfp4,
        .from_float_ref           = (ggml_from_float_t)quantize_row_nvfp4_ref,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = शून्य,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = शून्य,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = शून्य,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = शून्य,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_आकार                = QK4_NL,
        .type_आकार                = माप(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_q8_K),
        .is_quantized             = true,
    },
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_आकार                = 1,
        .type_आकार                = माप(ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
        .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_bf16_row_ref,
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
        .type_आकार                = माप(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_आकार                = QK_K,
        .type_आकार                = माप(block_tq2_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
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

स्थिर काष्ठा ggml_type_traits * ggml_get_type_traits(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस &type_traits[type];
}

//
// ggml object
//

काष्ठा ggml_object {
    माप_प्रकार offs;
    माप_प्रकार आकार;

    काष्ठा ggml_object * अग्र;

    क्रमागत ggml_object_type type;

    अक्षर padding[4];
};

अटल स्थिर माप_प्रकार GGML_OBJECT_आकार = माप(काष्ठा ggml_object);

//
// ggml context
//

काष्ठा ggml_context {
    माप_प्रकार mem_आकार;
    व्योम * mem_बफर;
    bool   mem_बफर_owned;
    bool   no_alloc;

    पूर्णांक    n_objects;

    काष्ठा ggml_object * objects_begin;
    काष्ठा ggml_object * objects_end;
};

//
// data types
//

अटल स्थिर अक्षर * GGML_OP_NAME[GGML_OP_गिनती] = {
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

static_assert(GGML_OP_गिनती == 102, "GGML_OP_COUNT != 102");

अटल स्थिर अक्षर * GGML_OP_प्रतीक[GGML_OP_गिनती] = {
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

static_assert(GGML_OP_गिनती == 102, "GGML_OP_COUNT != 102");

static_assert(GGML_OP_POOL_गिनती == 2, "GGML_OP_POOL_COUNT != 2");

अटल स्थिर अक्षर * GGML_UNARY_OP_NAME[GGML_UNARY_OP_गिनती] = {
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

static_assert(GGML_UNARY_OP_गिनती == 22, "GGML_UNARY_OP_COUNT != 22");

अटल स्थिर अक्षर * GGML_GLU_OP_NAME[GGML_GLU_OP_गिनती] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
    "SWIGLU_CLAMP",
};

static_assert(GGML_GLU_OP_गिनती == 7, "GGML_GLU_OP_COUNT != 7");

static_assert(माप(काष्ठा ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(माप(काष्ठा ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

व्योम ggml_print_object(स्थिर काष्ठा ggml_object * obj) {
    GGML_LOG_INFO(" - ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->आकार, (स्थिर व्योम *) obj->अग्र);
}

व्योम ggml_print_objects(स्थिर काष्ठा ggml_context * ctx) {
    काष्ठा ggml_object * obj = ctx->objects_begin;

    GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (स्थिर व्योम *) ctx);

    जबतक (obj != शून्य) {
        ggml_print_object(obj);
        obj = obj->अग्र;
    }

    GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t ggml_nrows(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

माप_प्रकार ggml_nbytes(स्थिर काष्ठा ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        अगर (tensor->ne[i] <= 0) {
            वापस 0;
        }
    }

    माप_प्रकार nbytes;
    स्थिर माप_प्रकार blck_आकार = ggml_blck_आकार(tensor->type);
    अगर (blck_आकार == 1) {
        nbytes = ggml_type_आकार(tensor->type);
        क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    अन्यथा {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_आकार;
        क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    वापस nbytes;
}

माप_प्रकार ggml_nbytes_pad(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

int64_t ggml_blck_आकार(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस type_traits[type].blck_आकार;
}

माप_प्रकार ggml_type_आकार(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस type_traits[type].type_आकार;
}

माप_प्रकार ggml_row_आकार(क्रमागत ggml_type type, int64_t ne) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    निश्चित(ne % ggml_blck_आकार(type) == 0);
    वापस ggml_type_आकार(type)*ne/ggml_blck_आकार(type);
}

द्विगुन ggml_type_sizef(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस ((द्विगुन)(type_traits[type].type_आकार))/type_traits[type].blck_आकार;
}

स्थिर अक्षर * ggml_type_name(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस type_traits[type].type_name;
}

bool ggml_is_quantized(क्रमागत ggml_type type) {
    निश्चित(type >= 0);
    निश्चित(type < GGML_TYPE_गिनती);
    वापस type_traits[type].is_quantized;
}

स्थिर अक्षर * ggml_op_name(क्रमागत ggml_op op) {
    वापस GGML_OP_NAME[op];
}

स्थिर अक्षर * ggml_op_प्रतीक(क्रमागत ggml_op op) {
    वापस GGML_OP_प्रतीक[op];
}

स्थिर अक्षर * ggml_unary_op_name(क्रमागत ggml_unary_op op) {
    वापस GGML_UNARY_OP_NAME[op];
}

स्थिर अक्षर * ggml_glu_op_name(क्रमागत ggml_glu_op op) {
    वापस GGML_GLU_OP_NAME[op];
}

स्थिर अक्षर * ggml_op_desc(स्थिर काष्ठा ggml_tensor * t) {
    अगर (t->op == GGML_OP_UNARY) {
        क्रमागत ggml_unary_op uop = ggml_get_unary_op(t);
        वापस ggml_unary_op_name(uop);
    }
    अगर (t->op == GGML_OP_GLU) {
        क्रमागत ggml_glu_op gop = ggml_get_glu_op(t);
        वापस ggml_glu_op_name(gop);
    }
    वापस ggml_op_name(t->op);
}

माप_प्रकार ggml_element_आकार(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_type_आकार(tensor->type);
}

bool ggml_is_scalar(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_vector(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_matrix(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool ggml_is_3d(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस tensor->ne[3] == 1;
}

पूर्णांक ggml_n_dims(स्थिर काष्ठा ggml_tensor * tensor) {
    क्रम (पूर्णांक i = GGML_MAX_DIMS - 1; i >= 1; --i) {
        अगर (tensor->ne[i] > 1) {
            वापस i + 1;
        }
    }
    वापस 1;
}

क्रमागत ggml_type ggml_ftype_to_ggml_type(क्रमागत ggml_ftype ftype) {
    क्रमागत ggml_type wtype = GGML_TYPE_गिनती;

    चयन (ftype) {
        हाल GGML_FTYPE_ALL_F32:              wtype = GGML_TYPE_F32;   अवरोध;
        हाल GGML_FTYPE_MOSTLY_F16:           wtype = GGML_TYPE_F16;   अवरोध;
        हाल GGML_FTYPE_MOSTLY_BF16:          wtype = GGML_TYPE_BF16;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q4_0:          wtype = GGML_TYPE_Q4_0;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q4_1:          wtype = GGML_TYPE_Q4_1;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q1_0:          wtype = GGML_TYPE_Q1_0;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q5_0:          wtype = GGML_TYPE_Q5_0;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q5_1:          wtype = GGML_TYPE_Q5_1;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q8_0:          wtype = GGML_TYPE_Q8_0;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_MXFP4:         wtype = GGML_TYPE_MXFP4; अवरोध;
        हाल GGML_FTYPE_MOSTLY_NVFP4:         wtype = GGML_TYPE_NVFP4; अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q2_K:          wtype = GGML_TYPE_Q2_K;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q3_K:          wtype = GGML_TYPE_Q3_K;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q4_K:          wtype = GGML_TYPE_Q4_K;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q5_K:          wtype = GGML_TYPE_Q5_K;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q6_K:          wtype = GGML_TYPE_Q6_K;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = GGML_TYPE_IQ2_XXS;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = GGML_TYPE_IQ2_XS;   अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = GGML_TYPE_IQ3_XXS;  अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ1_S:         wtype = GGML_TYPE_IQ1_S;    अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ1_M:         wtype = GGML_TYPE_IQ1_M;    अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = GGML_TYPE_IQ4_NL;   अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = GGML_TYPE_IQ4_XS;   अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ3_S:         wtype = GGML_TYPE_IQ3_S;    अवरोध;
        हाल GGML_FTYPE_MOSTLY_IQ2_S:         wtype = GGML_TYPE_IQ2_S;    अवरोध;
        हाल GGML_FTYPE_UNKNOWN:              wtype = GGML_TYPE_गिनती; अवरोध;
        हाल GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = GGML_TYPE_गिनती; अवरोध;
    }

    GGML_ASSERT(wtype != GGML_TYPE_गिनती);

    वापस wtype;
}

माप_प्रकार ggml_tensor_overhead(व्योम) {
    वापस GGML_OBJECT_आकार + GGML_TENSOR_आकार;
}

bool ggml_is_transposed(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस tensor->nb[0] > tensor->nb[1];
}

अटल bool ggml_is_contiguous_m_n(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक m, पूर्णांक n) {
    माप_प्रकार अग्र_nb = ggml_type_आकार(tensor->type);
    अगर (tensor->ne[0] != ggml_blck_आकार(tensor->type) && tensor->nb[0] != अग्र_nb) {
        वापस false;
    }
    अग्र_nb *= tensor->ne[0]/ggml_blck_आकार(tensor->type);
    क्रम (पूर्णांक i = 1; i < n; i++) {
        अगर (i > m) {
            अगर (tensor->ne[i] != 1 && tensor->nb[i] != अग्र_nb) {
                वापस false;
            }
            अग्र_nb *= tensor->ne[i];
        } अन्यथा {
            // this dimension does not need to be contiguous
            अग्र_nb = tensor->ne[i]*tensor->nb[i];
        }
    }
    वापस true;
}

bool ggml_is_contiguous(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_0(tensor);
}

bool ggml_is_contiguous_0(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 0, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_1(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 1, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_2(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 2, GGML_MAX_DIMS);
}

bool ggml_is_contiguous_to_1(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 0, 1);
}

bool ggml_is_contiguous_to_2(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 0, 2);
}

bool ggml_is_contiguous_to_3(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_is_contiguous_m_n(tensor, 0, 3);
}

bool ggml_is_contiguously_allocated(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_आकार(tensor->type)/ggml_blck_आकार(tensor->type);
}

bool ggml_is_permuted(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

bool ggml_is_contiguous_channels(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस
        tensor->nb[0] > tensor->nb[2] &&
        tensor->nb[1] > tensor->nb[0] &&
        tensor->nb[2] == ggml_type_आकार(tensor->type);
}

bool ggml_is_contiguous_rows(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस
        tensor->ne[0] == ggml_blck_आकार(tensor->type) ||
        tensor->nb[0] == ggml_type_आकार(tensor->type);
}

अटल अंतरभूत bool ggml_is_padded_1d(स्थिर काष्ठा ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस
        tensor->nb[0] == ggml_type_आकार(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool ggml_is_empty(स्थिर काष्ठा ggml_tensor * tensor) {
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; ++i) {
        अगर (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            वापस true;
        }
    }
    वापस false;
}

bool ggml_are_same_shape(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_are_same_stride(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

bool ggml_is_view(स्थिर काष्ठा ggml_tensor * t) {
    वापस ggml_impl_is_view(t);
}

// check if t1 can be represented as a repetition of t0
bool ggml_can_repeat(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस ggml_is_empty(t0) ? ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

अटल अंतरभूत bool ggml_can_repeat_rows(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#घोषणा GGML_ASSERT_ALIGNED(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

काष्ठा ggml_context * ggml_init(काष्ठा ggml_init_params params) {
    अटल bool is_first_call = true;

    ggml_critical_section_start();

    अगर (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        is_first_call = false;
    }

    ggml_critical_section_end();

    काष्ठा ggml_context * ctx = GGML_MALLOC(माप(काष्ठा ggml_context));

    // allow to call ggml_init with 0 size
    अगर (params.mem_आकार == 0) {
        params.mem_आकार = GGML_MEM_ALIGN;
    }

    स्थिर माप_प्रकार mem_आकार = params.mem_बफर ? params.mem_आकार : GGML_PAD(params.mem_आकार, GGML_MEM_ALIGN);

    *ctx = (काष्ठा ggml_context) {
        /*.mem_size           =*/ mem_आकार,
        /*.mem_buffer         =*/ params.mem_बफर ? params.mem_बफर : ggml_aligned_malloc(mem_आकार),
        /*.mem_buffer_owned   =*/ params.mem_बफर ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ शून्य,
        /*.objects_end        =*/ शून्य,
    };

    GGML_ASSERT(ctx->mem_बफर != शून्य);

    GGML_ASSERT_ALIGNED(ctx->mem_बफर);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    वापस ctx;
}

व्योम ggml_reset(काष्ठा ggml_context * ctx) {
    अगर (ctx == शून्य) {
        वापस;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = शून्य;
    ctx->objects_end   = शून्य;
}

व्योम ggml_free(काष्ठा ggml_context * ctx) {
    अगर (ctx == शून्य) {
        वापस;
    }

    अगर (ctx->mem_बफर_owned) {
        ggml_aligned_free(ctx->mem_बफर, ctx->mem_आकार);
    }

    GGML_FREE(ctx);
}

माप_प्रकार ggml_used_mem(स्थिर काष्ठा ggml_context * ctx) {
    वापस ctx->objects_end == शून्य ? 0 : ctx->objects_end->offs + ctx->objects_end->आकार;
}

bool ggml_get_no_alloc(काष्ठा ggml_context * ctx) {
    वापस ctx->no_alloc;
}

व्योम ggml_set_no_alloc(काष्ठा ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

व्योम * ggml_get_mem_बफर(स्थिर काष्ठा ggml_context * ctx) {
    वापस ctx->mem_बफर;
}

माप_प्रकार ggml_get_mem_आकार(स्थिर काष्ठा ggml_context * ctx) {
    वापस ctx->mem_आकार;
}

माप_प्रकार ggml_get_max_tensor_आकार(स्थिर काष्ठा ggml_context * ctx) {
    माप_प्रकार max_आकार = 0;

    क्रम (काष्ठा ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor != शून्य; tensor = ggml_get_अग्र_tensor(ctx, tensor)) {
        माप_प्रकार bytes = ggml_nbytes(tensor);
        max_आकार = MAX(max_आकार, bytes);
    }

    वापस max_आकार;
}

////////////////////////////////////////////////////////////////////////////////

अटल काष्ठा ggml_object * ggml_new_object(काष्ठा ggml_context * ctx, क्रमागत ggml_object_type type, माप_प्रकार आकार) {
    // always insert objects at the end of the context's memory pool
    काष्ठा ggml_object * obj_cur = ctx->objects_end;

    स्थिर माप_प्रकार cur_offs = obj_cur == शून्य ? 0 : obj_cur->offs;
    स्थिर माप_प्रकार cur_आकार = obj_cur == शून्य ? 0 : obj_cur->आकार;
    स्थिर माप_प्रकार cur_end  = cur_offs + cur_आकार;

    // align to GGML_MEM_ALIGN
    GGML_ASSERT(आकार <= SIZE_MAX - (GGML_MEM_ALIGN - 1));
    माप_प्रकार आकार_needed = GGML_PAD(आकार, GGML_MEM_ALIGN);

    अक्षर * स्थिर mem_बफर = ctx->mem_बफर;
    काष्ठा ggml_object * स्थिर obj_new = (काष्ठा ggml_object *)(mem_बफर + cur_end);

    // integer overflow checks
    अगर (cur_end > SIZE_MAX - आकार_needed) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu)\n", __func__, cur_end, आकार_needed);
        वापस शून्य;
    }
    अगर (cur_end + आकार_needed > SIZE_MAX - GGML_OBJECT_आकार) {
        GGML_LOG_WARN("%s: overflow detected in cur_end (%zu) + size_needed (%zu) + GGML_OBJECT_SIZE (%zu)\n", __func__,
                cur_end, आकार_needed, (माप_प्रकार) GGML_OBJECT_आकार);
        वापस शून्य;
    }

    अगर (cur_end + आकार_needed + GGML_OBJECT_आकार > ctx->mem_आकार) {
        GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + आकार_needed + GGML_OBJECT_आकार, ctx->mem_आकार);
#अगर_अघोषित न_संशोधन
        GGML_ABORT("not enough space in the context's memory pool");
#पूर्ण_अगर
        वापस शून्य;
    }

    *obj_new = (काष्ठा ggml_object) {
        .offs = cur_end + GGML_OBJECT_आकार,
        .आकार = आकार_needed,
        .अग्र = शून्य,
        .type = type,
    };

    GGML_ASSERT_ALIGNED(mem_बफर + obj_new->offs);

    अगर (obj_cur != शून्य) {
        obj_cur->अग्र = obj_new;
    } अन्यथा {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    वापस obj_new;
}

अटल काष्ठा ggml_tensor * ggml_new_tensor_impl(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type      type,
        पूर्णांक                   n_dims,
        स्थिर int64_t       * ne,
        काष्ठा ggml_tensor  * view_src,
        माप_प्रकार                view_offs) {

    GGML_ASSERT(type >= 0 && type < GGML_TYPE_गिनती);
    GGML_ASSERT(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    अगर (view_src != शून्य && view_src->view_src != शून्य) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    // validate number of elements to fit in int64_t
    int64_t current_nelements = ne[0];
    क्रम (पूर्णांक i = 1; i < n_dims; i++) {
        अगर (ne[i] > 1) {
            GGML_ASSERT(INT64_MAX / ne[i] > current_nelements);
            current_nelements *= ne[i];
        }
    }

    माप_प्रकार तथ्य_आकार = ggml_row_आकार(type, ne[0]);
    क्रम (पूर्णांक i = 1; i < n_dims; i++) {
        तथ्य_आकार *= ne[i];
    }

    GGML_ASSERT(view_src == शून्य || तथ्य_आकार == 0 || तथ्य_आकार + view_offs <= ggml_nbytes(view_src));

    व्योम * तथ्य = view_src != शून्य ? view_src->तथ्य : शून्य;
    अगर (तथ्य != शून्य) {
        तथ्य = (अक्षर *) तथ्य + view_offs;
    }

    माप_प्रकार obj_alloc_आकार = 0;

    अगर (view_src == शून्य && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_आकार = तथ्य_आकार;
    }

    GGML_ASSERT(GGML_TENSOR_आकार <= SIZE_MAX - obj_alloc_आकार);

    काष्ठा ggml_object * स्थिर obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_आकार + obj_alloc_आकार);
    GGML_ASSERT(obj_new);

    काष्ठा ggml_tensor * स्थिर परिणाम = (काष्ठा ggml_tensor *)((अक्षर *)ctx->mem_बफर + obj_new->offs);

    *परिणाम = (काष्ठा ggml_tensor) {
        /*.type         =*/ type,
        /*.buffer       =*/ शून्य,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.src          =*/ { शून्य },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_आकार > 0 ? (व्योम *)(परिणाम + 1) : तथ्य,
        /*.name         =*/ { 0 },
        /*.extra        =*/ शून्य,
        /*.padding      =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //GGML_ASSERT_ALIGNED(result->data);

    क्रम (पूर्णांक i = 0; i < n_dims; i++) {
        परिणाम->ne[i] = ne[i];
    }

    परिणाम->nb[0] = ggml_type_आकार(type);
    परिणाम->nb[1] = परिणाम->nb[0]*(परिणाम->ne[0]/ggml_blck_आकार(type));
    क्रम (पूर्णांक i = 2; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = परिणाम->nb[i - 1]*परिणाम->ne[i - 1];
    }

    ctx->n_objects++;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_new_tensor(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type      type,
        पूर्णांक                   n_dims,
        स्थिर int64_t       * ne) {
    वापस ggml_new_tensor_impl(ctx, type, n_dims, ne, शून्य, 0);
}

काष्ठा ggml_tensor * ggml_new_tensor_1d(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type      type,
        int64_t ne0) {
    वापस ggml_new_tensor(ctx, type, 1, &ne0);
}

काष्ठा ggml_tensor * ggml_new_tensor_2d(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    स्थिर int64_t ne[2] = { ne0, ne1 };
    वापस ggml_new_tensor(ctx, type, 2, ne);
}

काष्ठा ggml_tensor * ggml_new_tensor_3d(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    स्थिर int64_t ne[3] = { ne0, ne1, ne2 };
    वापस ggml_new_tensor(ctx, type, 3, ne);
}

काष्ठा ggml_tensor * ggml_new_tensor_4d(
        काष्ठा ggml_context * ctx,
        क्रमागत   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    स्थिर int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    वापस ggml_new_tensor(ctx, type, 4, ne);
}

व्योम * ggml_new_बफर(काष्ठा ggml_context * ctx, माप_प्रकार nbytes) {
    काष्ठा ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_बफर, nbytes);

    वापस (uint8_t *)ctx->mem_बफर + obj->offs;
}

काष्ठा ggml_tensor * ggml_dup_tensor(काष्ठा ggml_context * ctx, स्थिर काष्ठा ggml_tensor * src) {
    वापस ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
}

व्योम ggml_unravel_सूचकांक(स्थिर काष्ठा ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    स्थिर int64_t ne2 = tensor->ne[2];
    स्थिर int64_t ne1 = tensor->ne[1];
    स्थिर int64_t ne0 = tensor->ne[0];

    स्थिर int64_t i3_ = (i/(ne2*ne1*ne0));
    स्थिर int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    स्थिर int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    स्थिर int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    अगर (i0) {
        * i0 = i0_;
    }
    अगर (i1) {
        * i1 = i1_;
    }
    अगर (i2) {
        * i2 = i2_;
    }
    अगर (i3) {
        * i3 = i3_;
    }
}

व्योम * ggml_get_तथ्य(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस tensor->तथ्य;
}

भग्न * ggml_get_तथ्य_f32(स्थिर काष्ठा ggml_tensor * tensor) {
    निश्चित(tensor->type == GGML_TYPE_F32);
    वापस (भग्न *)(tensor->तथ्य);
}

क्रमागत ggml_unary_op ggml_get_unary_op(स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_UNARY);
    वापस (क्रमागत ggml_unary_op) ggml_get_op_params_i32(tensor, 0);
}

क्रमागत ggml_glu_op ggml_get_glu_op(स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_GLU);
    वापस (क्रमागत ggml_glu_op) ggml_get_op_params_i32(tensor, 0);
}

स्थिर अक्षर * ggml_get_name(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस tensor->name;
}

काष्ठा ggml_tensor * ggml_set_name(काष्ठा ggml_tensor * tensor, स्थिर अक्षर * name) {
    माप_प्रकार i;
    क्रम (i = 0; i < माप(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    वापस tensor;
}

काष्ठा ggml_tensor * ggml_format_name(काष्ठा ggml_tensor * tensor, स्थिर अक्षर * fmt, ...) {
    बहु_सूची args;
    बहु_शुरू(args, fmt);
    vsnprintf(tensor->name, माप(tensor->name), fmt, args);
    बहु_पूर्ण(args);
    वापस tensor;
}

काष्ठा ggml_tensor * ggml_view_tensor(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * src) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, src->type, GGML_MAX_DIMS, src->ne, src, 0);
    ggml_format_name(परिणाम, "%s (view)", src->name);

    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        परिणाम->nb[i] = src->nb[i];
    }

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_get_first_tensor(स्थिर काष्ठा ggml_context * ctx) {
    काष्ठा ggml_object * obj = ctx->objects_begin;

    अक्षर * स्थिर mem_बफर = ctx->mem_बफर;

    जबतक (obj != शून्य) {
        अगर (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            वापस (काष्ठा ggml_tensor *)(mem_बफर + obj->offs);
        }

        obj = obj->अग्र;
    }

    वापस शून्य;
}

काष्ठा ggml_tensor * ggml_get_अग्र_tensor(स्थिर काष्ठा ggml_context * ctx, काष्ठा ggml_tensor * tensor) {
    काष्ठा ggml_object * obj = (काष्ठा ggml_object *) ((अक्षर *)tensor - GGML_OBJECT_आकार);
    obj = obj->अग्र;

    अक्षर * स्थिर mem_बफर = ctx->mem_बफर;

    जबतक (obj != शून्य) {
        अगर (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            वापस (काष्ठा ggml_tensor *)(mem_बफर + obj->offs);
        }

        obj = obj->अग्र;
    }

    वापस शून्य;
}

काष्ठा ggml_tensor * ggml_get_tensor(काष्ठा ggml_context * ctx, स्थिर अक्षर * name) {
    काष्ठा ggml_object * obj = ctx->objects_begin;

    अक्षर * स्थिर mem_बफर = ctx->mem_बफर;

    जबतक (obj != शून्य) {
        अगर (obj->type == GGML_OBJECT_TYPE_TENSOR) {
            काष्ठा ggml_tensor * cur = (काष्ठा ggml_tensor *)(mem_बफर + obj->offs);
            अगर (म_भेद(cur->name, name) == 0) {
                वापस cur;
            }
        }

        obj = obj->अग्र;
    }

    वापस शून्य;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

अटल काष्ठा ggml_tensor * ggml_dup_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_DUP;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_dup(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_dup_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_dup_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_dup_impl(ctx, a, true);
}

// ggml_add

अटल काष्ठा ggml_tensor * ggml_add_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_ADD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_add(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_add_impl(ctx, a, b, false);
}

काष्ठा ggml_tensor * ggml_add_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_add_impl(ctx, a, b, true);
}

// ggml_add_cast

अटल काष्ठा ggml_tensor * ggml_add_cast_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        क्रमागत   ggml_type      type) {
    // TODO: support less-strict constraint
    //       GGML_ASSERT(ggml_can_repeat(b, a));
    GGML_ASSERT(ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    GGML_ASSERT(ggml_is_quantized(a->type) ||
                a->type == GGML_TYPE_F16 ||
                a->type == GGML_TYPE_BF16);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);

    परिणाम->op     = GGML_OP_ADD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_add_cast(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        क्रमागत   ggml_type      type) {
    वापस ggml_add_cast_impl(ctx, a, b, type);
}

काष्ठा ggml_tensor * ggml_add_id(
            काष्ठा ggml_context * ctx,
            काष्ठा ggml_tensor  * a,
            काष्ठा ggml_tensor  * b,
            काष्ठा ggml_tensor  * ids) {

    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->ne[1] == ids->ne[0]);
    GGML_ASSERT(a->ne[2] == ids->ne[1]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_ADD_ID;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = ids;

    वापस परिणाम;
}

// ggml_add1

अटल काष्ठा ggml_tensor * ggml_add1_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_ADD1;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_add1(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_add1_impl(ctx, a, b, false);
}

काष्ठा ggml_tensor * ggml_add1_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_add1_impl(ctx, a, b, true);
}

// ggml_acc

अटल काष्ठा ggml_tensor * ggml_acc_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(b) <= ggml_nelements(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (माप_प्रकार)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_ACC;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_acc(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset) {
    वापस ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

काष्ठा ggml_tensor * ggml_acc_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset) {
    वापस ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// ggml_sub

अटल काष्ठा ggml_tensor * ggml_sub_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SUB;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_sub(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_sub_impl(ctx, a, b, false);
}

काष्ठा ggml_tensor * ggml_sub_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

अटल काष्ठा ggml_tensor * ggml_mul_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_MUL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_mul(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_mul_impl(ctx, a, b, false);
}

काष्ठा ggml_tensor * ggml_mul_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

अटल काष्ठा ggml_tensor * ggml_div_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  inplace) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_DIV;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_div(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_div_impl(ctx, a, b, false);
}

काष्ठा ggml_tensor * ggml_div_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

अटल काष्ठा ggml_tensor * ggml_sqr_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SQR;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_sqr(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sqr_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_sqr_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

अटल काष्ठा ggml_tensor * ggml_sqrt_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SQRT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_sqrt(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sqrt_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_sqrt_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sqrt_impl(ctx, a, true);
}

// ggml_log

अटल काष्ठा ggml_tensor * ggml_log_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_LOG;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_log(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_log_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_log_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_log_impl(ctx, a, true);
}

काष्ठा ggml_tensor * ggml_expm1(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_EXPM1);
}

काष्ठा ggml_tensor * ggml_expm1_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXPM1);
}

काष्ठा ggml_tensor * ggml_softplus(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

काष्ठा ggml_tensor * ggml_softplus_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SOFTPLUS);
}

// ggml_sin

अटल काष्ठा ggml_tensor * ggml_sin_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SIN;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_sin(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sin_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_sin_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_sin_impl(ctx, a, true);
}

// ggml_cos

अटल काष्ठा ggml_tensor * ggml_cos_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_COS;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_cos(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_cos_impl(ctx, a, false);
}

काष्ठा ggml_tensor * ggml_cos_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_cos_impl(ctx, a, true);
}

// ggml_sum

काष्ठा ggml_tensor * ggml_योग(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, a->type, 1);

    परिणाम->op     = GGML_OP_योग;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_sum_rows

काष्ठा ggml_tensor * ggml_योग_rows(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    int64_t ne[GGML_MAX_DIMS] = { 1 };
    क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    परिणाम->op     = GGML_OP_योग_ROWS;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_cumsum

काष्ठा ggml_tensor * ggml_cumsum(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_CUMSUM;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_mean

काष्ठा ggml_tensor * ggml_mean(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_MEAN;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_argmax

काष्ठा ggml_tensor * ggml_argmax(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, a->ne[1]);

    परिणाम->op     = GGML_OP_ARGMAX;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_count_equal

काष्ठा ggml_tensor * ggml_गिनती_equal(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);

    परिणाम->op     = GGML_OP_गिनती_EQUAL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_repeat

काष्ठा ggml_tensor * ggml_repeat(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    परिणाम->op     = GGML_OP_REPEAT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_repeat_4d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor * a,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    स्थिर bool can_repeat = ggml_is_empty(a) || (
        (ne0 % a->ne[0] == 0) &&
        (ne1 % a->ne[1] == 0) &&
        (ne2 % a->ne[2] == 0) &&
        (ne3 % a->ne[3] == 0)
    );
    GGML_ASSERT(can_repeat);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    परिणाम->op     = GGML_OP_REPEAT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_repeat_back

काष्ठा ggml_tensor * ggml_repeat_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_repeat(b, a));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, b->ne);

    परिणाम->op     = GGML_OP_REPEAT_BACK;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_concat

काष्ठा ggml_tensor * ggml_concat(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    काष्ठा ggml_tensor  * b,
    पूर्णांक                   dim) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(a->type == b->type);

    int64_t ne[GGML_MAX_DIMS];
    क्रम (पूर्णांक d = 0; d < GGML_MAX_DIMS; ++d) {
        अगर (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            जारी;
        }
        GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, GGML_MAX_DIMS, ne);

    ggml_set_op_params_i32(परिणाम, 0, dim);

    परिणाम->op     = GGML_OP_CONCAT;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_abs

काष्ठा ggml_tensor * ggml_abs(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
}

काष्ठा ggml_tensor * ggml_abs_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ABS);
}

// ggml_sgn

काष्ठा ggml_tensor * ggml_sgn(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_SGN);
}

काष्ठा ggml_tensor * ggml_sgn_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SGN);
}

// ggml_neg

काष्ठा ggml_tensor * ggml_neg(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_NEG);
}

काष्ठा ggml_tensor * ggml_neg_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_NEG);
}

// ggml_step

काष्ठा ggml_tensor * ggml_step(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_STEP);
}

काष्ठा ggml_tensor * ggml_step_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_STEP);
}

// ggml_tanh

काष्ठा ggml_tensor * ggml_tanh(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_TANH);
}

काष्ठा ggml_tensor * ggml_tanh_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TANH);
}

// ggml_elu

काष्ठा ggml_tensor * ggml_elu(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_ELU);
}

काष्ठा ggml_tensor * ggml_elu_inplace(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ELU);
}

// ggml_relu

काष्ठा ggml_tensor * ggml_relu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_RELU);
}

काष्ठा ggml_tensor * ggml_relu_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_RELU);
}

// ggml_leaky_relu

काष्ठा ggml_tensor * ggml_leaky_relu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 negative_slope,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(परिणाम, &negative_slope, माप(negative_slope));

    परिणाम->op     = GGML_OP_LEAKY_RELU;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_sigmoid

काष्ठा ggml_tensor * ggml_sigmoid(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_SIGMOID);
}

काष्ठा ggml_tensor * ggml_sigmoid_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SIGMOID);
}

// ggml_gelu

काष्ठा ggml_tensor * ggml_gelu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_GELU);
}

काष्ठा ggml_tensor * ggml_gelu_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU);
}

// ggml_gelu_erf

काष्ठा ggml_tensor * ggml_gelu_erf(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

काष्ठा ggml_tensor * ggml_gelu_erf_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_ERF);
}

// ggml_gelu_quick

काष्ठा ggml_tensor * ggml_gelu_quick(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

काष्ठा ggml_tensor * ggml_gelu_quick_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_GELU_QUICK);
}

// ggml_silu

काष्ठा ggml_tensor * ggml_silu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

काष्ठा ggml_tensor * ggml_silu_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_SILU);
}

// ggml_xielu

काष्ठा ggml_tensor * ggml_xielu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न alpha_n,
        भग्न alpha_p,
        भग्न beta,
        भग्न eps) {
    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) GGML_UNARY_OP_XIELU);
    ggml_set_op_params_f32(परिणाम, 1, beta + ggml_compute_softplus_f32(alpha_n));
    ggml_set_op_params_f32(परिणाम, 2, ggml_compute_softplus_f32(alpha_p));
    ggml_set_op_params_f32(परिणाम, 3, beta);
    ggml_set_op_params_f32(परिणाम, 4, eps);

    परिणाम->op     = GGML_OP_UNARY;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_silu_back

काष्ठा ggml_tensor * ggml_silu_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SILU_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml hardswish

काष्ठा ggml_tensor * ggml_hardswish(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

काष्ठा ggml_tensor * ggml_hardsigmoid(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

काष्ठा ggml_tensor * ggml_exp(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_EXP);
}

काष्ठा ggml_tensor * ggml_exp_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_EXP);
}

// ggml_glu

अटल काष्ठा ggml_tensor * ggml_glu_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        क्रमागत ggml_glu_op      op,
        bool                  swapped) {
    GGML_ASSERT(ggml_is_contiguous_1(a));

    अगर (b) {
        GGML_ASSERT(ggml_is_contiguous_1(b));
        GGML_ASSERT(ggml_are_same_shape(a, b));
        GGML_ASSERT(a->type == b->type);
    }

    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 }; क्रम (पूर्णांक i = 1; i < GGML_MAX_DIMS; i++) ne[i] = a->ne[i];
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b ? a->ne : ne, शून्य, 0);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) op);
    ggml_set_op_params_i32(परिणाम, 1, (int32_t) swapped);

    परिणाम->op     = GGML_OP_GLU;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_floor

काष्ठा ggml_tensor * ggml_floor(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_FLOOR);
}

काष्ठा ggml_tensor * ggml_floor_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_FLOOR);
}

// ggml_ceil

काष्ठा ggml_tensor * ggml_ceil(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_CEIL);
}

काष्ठा ggml_tensor * ggml_ceil_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_CEIL);
}

//ggml_round

काष्ठा ggml_tensor * ggml_round(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_ROUND);
}

काष्ठा ggml_tensor * ggml_round_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_ROUND);
}

//ggml_trunc

काष्ठा ggml_tensor * ggml_trunc(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary(ctx, a, GGML_UNARY_OP_TRUNC);
}

काष्ठा ggml_tensor * ggml_trunc_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_unary_inplace(ctx, a, GGML_UNARY_OP_TRUNC);
}

काष्ठा ggml_tensor * ggml_glu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_glu_op      op,
        bool                  swapped) {
    वापस ggml_glu_impl(ctx, a, शून्य, op, swapped);
}

काष्ठा ggml_tensor * ggml_glu_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        क्रमागत ggml_glu_op      op) {
    वापस ggml_glu_impl(ctx, a, b, op, false);
}

// ggml_reglu

काष्ठा ggml_tensor * ggml_reglu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_REGLU, false);
}

काष्ठा ggml_tensor * ggml_reglu_swapped(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_REGLU, true);
}

काष्ठा ggml_tensor * ggml_reglu_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_glu_impl(ctx, a, b, GGML_GLU_OP_REGLU, false);
}

// ggml_geglu

काष्ठा ggml_tensor * ggml_geglu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU, false);
}

काष्ठा ggml_tensor * ggml_geglu_swapped(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU, true);
}

काष्ठा ggml_tensor * ggml_geglu_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU, false);
}

// ggml_swiglu

काष्ठा ggml_tensor * ggml_swiglu(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_SWIGLU, false);
}

काष्ठा ggml_tensor * ggml_swiglu_swapped(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_SWIGLU, true);
}

काष्ठा ggml_tensor * ggml_swiglu_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU, false);
}

// ggml_geglu_erf

काष्ठा ggml_tensor * ggml_geglu_erf(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU_ERF, false);
}

काष्ठा ggml_tensor * ggml_geglu_erf_swapped(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU_ERF, true);
}

काष्ठा ggml_tensor * ggml_geglu_erf_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_ERF, false);
}

// ggml_geglu_quick

काष्ठा ggml_tensor * ggml_geglu_quick(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU_QUICK, false);
}

काष्ठा ggml_tensor * ggml_geglu_quick_swapped(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_glu_impl(ctx, a, शून्य, GGML_GLU_OP_GEGLU_QUICK, true);
}

काष्ठा ggml_tensor * ggml_geglu_quick_split(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_glu_impl(ctx, a, b, GGML_GLU_OP_GEGLU_QUICK, false);
}

काष्ठा ggml_tensor * ggml_swiglu_oai(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 alpha,
        भग्न                 limit) {
    काष्ठा ggml_tensor * परिणाम = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_OAI, false);
    ggml_set_op_params_f32(परिणाम, 2, alpha);
    ggml_set_op_params_f32(परिणाम, 3, limit);

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_swiglu_clamp(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 limit) {
    काष्ठा ggml_tensor * परिणाम = ggml_glu_impl(ctx, a, b, GGML_GLU_OP_SWIGLU_CLAMP, false);
    ggml_set_op_params_f32(परिणाम, 3, limit);

    वापस परिणाम;
}

// ggml_norm

अटल काष्ठा ggml_tensor * ggml_norm_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(परिणाम, &eps, माप(eps));

    परिणाम->op     = GGML_OP_NORM;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_norm(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_norm_impl(ctx, a, eps, false);
}

काष्ठा ggml_tensor * ggml_norm_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm

अटल काष्ठा ggml_tensor * ggml_rms_norm_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params(परिणाम, &eps, माप(eps));

    परिणाम->op     = GGML_OP_RMS_NORM;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_rms_norm(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_rms_norm_impl(ctx, a, eps, false);
}

काष्ठा ggml_tensor * ggml_rms_norm_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_rms_norm_impl(ctx, a, eps, true);
}

// ggml_rms_norm_back

काष्ठा ggml_tensor * ggml_rms_norm_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 eps) {
    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(परिणाम, &eps, माप(eps));

    परिणाम->op     = GGML_OP_RMS_NORM_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_group_norm

अटल काष्ठा ggml_tensor * ggml_group_norm_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_groups,
        भग्न                 eps,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(परिणाम, 0, n_groups);
    ggml_set_op_params_f32(परिणाम, 1, eps);

    परिणाम->op     = GGML_OP_GROUP_NORM;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_group_norm(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_groups,
        भग्न                 eps) {
    वापस ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

काष्ठा ggml_tensor * ggml_group_norm_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_groups,
        भग्न                 eps) {
    वापस ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// ggml_l2_norm

अटल काष्ठा ggml_tensor * ggml_l2_norm_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(परिणाम, 0, eps);

    परिणाम->op     = GGML_OP_L2_NORM;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_l2_norm(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_l2_norm_impl(ctx, a, eps, false);
}

काष्ठा ggml_tensor * ggml_l2_norm_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 eps) {
    वापस ggml_l2_norm_impl(ctx, a, eps, true);
}

// ggml_prec

bool ggml_prec_set_acc(
        काष्ठा ggml_tensor * a,
        क्रमागत ggml_prec       prec) {
    चयन (a->op) {
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
            {
                स्थिर int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 0, prec_i32);
            }
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            {
                स्थिर int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 3, prec_i32);
            }
            अवरोध;
        शेष:
            वापस false;
    };

    वापस true;
}

bool ggml_prec_set_src(
        काष्ठा ggml_tensor * a,
        क्रमागत ggml_prec       prec,
        पूर्णांक                  idx) {
    GGML_ASSERT(idx >= 0 && idx < GGML_MAX_SRC);

    चयन (a->op) {
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
            {
                अगर (idx != 1) {
                    वापस false;
                }

                स्थिर int32_t prec_i32 = (int32_t) prec;
                ggml_set_op_params_i32(a, 2 + idx, prec_i32);
            }
            अवरोध;
        शेष:
            वापस false;
    };

    वापस true;
}

// ggml_mul_mat

अटल अंतरभूत bool ggml_can_mul_mat(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

काष्ठा ggml_tensor * ggml_mul_mat(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    स्थिर int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_MUL_MAT;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

व्योम ggml_mul_mat_set_prec(
        काष्ठा ggml_tensor * a,
        क्रमागत ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    स्थिर int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 0, prec_i32);
}

व्योम ggml_mul_mat_set_hint(
        काष्ठा ggml_tensor * a,
        क्रमागत ggml_op_hint    hint) {
    GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

    स्थिर int32_t hint_i32 = (int32_t) hint;

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
काष्ठा ggml_tensor * ggml_mul_mat_id(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * as,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * ids) {
    GGML_ASSERT(!ggml_is_transposed(as));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    GGML_ASSERT(b->ne[3] == 1); // b is 3d
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    स्थिर int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_MUL_MAT_ID;
    परिणाम->src[0] = as;
    परिणाम->src[1] = b;
    परिणाम->src[2] = ids;

    वापस परिणाम;
}

// ggml_out_prod

अटल अंतरभूत bool ggml_can_out_prod(स्थिर काष्ठा ggml_tensor * t0, स्थिर काष्ठा ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

काष्ठा ggml_tensor * ggml_out_prod(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_out_prod(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    स्थिर int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_OUT_PROD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_scale

अटल काष्ठा ggml_tensor * ggml_scale_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 s,
        भग्न                 b,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_padded_1d(a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भग्न params[2] = { s, b };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op     = GGML_OP_SCALE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_scale(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 s) {
    वापस ggml_scale_impl(ctx, a, s, 0.0, false);
}

काष्ठा ggml_tensor * ggml_scale_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 s) {
    वापस ggml_scale_impl(ctx, a, s, 0.0, true);
}

काष्ठा ggml_tensor * ggml_scale_bias(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 s,
        भग्न                 b) {
    वापस ggml_scale_impl(ctx, a, s, b, false);
}

काष्ठा ggml_tensor * ggml_scale_bias_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 s,
        भग्न                 b) {
    वापस ggml_scale_impl(ctx, a, s, b, true);
}

// ggml_set

अटल काष्ठा ggml_tensor * ggml_set_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset,
        bool                  inplace) {
    GGML_ASSERT(ggml_nelements(a) >= ggml_nelements(b));

    // make a view of the destination
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    GGML_ASSERT(offset < (माप_प्रकार)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_SET;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_set(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

काष्ठा ggml_tensor * ggml_set_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

काष्ठा ggml_tensor * ggml_set_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

काष्ठा ggml_tensor * ggml_set_1d_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

काष्ठा ggml_tensor * ggml_set_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

काष्ठा ggml_tensor * ggml_set_2d_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        माप_प्रकार                nb1,
        माप_प्रकार                offset) {
    वापस ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// ggml_cpy

अटल काष्ठा ggml_tensor * ggml_cpy_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    // make a view of the destination
    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, b);
    अगर (म_माप(b->name) > 0) {
        ggml_format_name(परिणाम, "%s (copy of %s)", b->name, a->name);
    } अन्यथा {
        ggml_format_name(परिणाम, "%s (copy)", a->name);
    }

    परिणाम->op     = GGML_OP_CPY;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_cpy(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor * a,
        काष्ठा ggml_tensor * b) {
    वापस ggml_cpy_impl(ctx, a, b);
}

काष्ठा ggml_tensor * ggml_cast(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत   ggml_type      type) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
    ggml_format_name(परिणाम, "%s (copy)", a->name);

    परिणाम->op     = GGML_OP_CPY;
    परिणाम->src[0] = a;
    परिणाम->src[1] = परिणाम; // note: this self-reference might seem redundant, but it's actually needed by some
                             //       backends for consistency with ggml_cpy_impl() above

    वापस परिणाम;
}

// ggml_cont

अटल काष्ठा ggml_tensor * ggml_cont_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (cont)", a->name);

    परिणाम->op     = GGML_OP_CONT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_cont(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor * a) {
    वापस ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
GGML_API काष्ठा ggml_tensor * ggml_cont_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0) {
    वापस ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API काष्ठा ggml_tensor * ggml_cont_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    वापस ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API काष्ठा ggml_tensor * ggml_cont_3d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    वापस ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

काष्ठा ggml_tensor * ggml_cont_4d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    ggml_format_name(परिणाम, "%s (cont)", a->name);

    परिणाम->op     = GGML_OP_CONT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_reshape

काष्ठा ggml_tensor * ggml_reshape(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor * a,
        काष्ठा ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS, b->ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_reshape_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0);

    स्थिर int64_t ne[1] = { ne0 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_reshape_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    स्थिर int64_t ne[2] = { ne0, ne1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_reshape_3d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    स्थिर int64_t ne[3] = { ne0, ne1, ne2 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_reshape_4d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    स्थिर int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    ggml_format_name(परिणाम, "%s (reshaped)", a->name);

    परिणाम->op     = GGML_OP_RESHAPE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

अटल काष्ठा ggml_tensor * ggml_view_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_dims,
        स्थिर int64_t       * ne,
        माप_प्रकार                offset) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(परिणाम, "%s (view)", a->name);

    ggml_set_op_params(परिणाम, &offset, माप(offset));

    परिणाम->op     = GGML_OP_VIEW;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_view_1d

काष्ठा ggml_tensor * ggml_view_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        माप_प्रकार                offset) {
    काष्ठा ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 1, &ne0, offset);

    वापस परिणाम;
}

// ggml_view_2d

काष्ठा ggml_tensor * ggml_view_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        माप_प्रकार                nb1,
        माप_प्रकार                offset) {
    स्थिर int64_t ne[2] = { ne0, ne1 };

    काष्ठा ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 2, ne, offset);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = परिणाम->nb[1]*ne1;
    परिणाम->nb[3] = परिणाम->nb[2];

    वापस परिणाम;
}

// ggml_view_3d

काष्ठा ggml_tensor * ggml_view_3d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                offset) {
    स्थिर int64_t ne[3] = { ne0, ne1, ne2 };

    काष्ठा ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 3, ne, offset);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = nb2;
    परिणाम->nb[3] = परिणाम->nb[2]*ne2;

    वापस परिणाम;
}

// ggml_view_4d

काष्ठा ggml_tensor * ggml_view_4d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        माप_प्रकार                nb1,
        माप_प्रकार                nb2,
        माप_प्रकार                nb3,
        माप_प्रकार                offset) {
    स्थिर int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    काष्ठा ggml_tensor * परिणाम = ggml_view_impl(ctx, a, 4, ne, offset);

    परिणाम->nb[1] = nb1;
    परिणाम->nb[2] = nb2;
    परिणाम->nb[3] = nb3;

    वापस परिणाम;
}

// ggml_permute

काष्ठा ggml_tensor * ggml_permute(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   axis0,
        पूर्णांक                   axis1,
        पूर्णांक                   axis2,
        पूर्णांक                   axis3) {
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

    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (permuted)", a->name);

    int64_t ne[GGML_MAX_DIMS];
    माप_प्रकार nb[GGML_MAX_DIMS];

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
    ggml_set_op_params(परिणाम, params, माप(params));

    वापस परिणाम;
}

// ggml_transpose

काष्ठा ggml_tensor * ggml_transpose(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);
    ggml_format_name(परिणाम, "%s (transposed)", a->name);

    परिणाम->ne[0] = a->ne[1];
    परिणाम->ne[1] = a->ne[0];

    परिणाम->nb[0] = a->nb[1];
    परिणाम->nb[1] = a->nb[0];

    परिणाम->op     = GGML_OP_TRANSPOSE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_get_rows

काष्ठा ggml_tensor * ggml_get_rows(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    GGML_ASSERT(b->ne[3] == 1);
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    // TODO: implement non F32 return
    क्रमागत ggml_type type = GGML_TYPE_F32;
    अगर (a->type == GGML_TYPE_I32) {
        type = a->type;
    }
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    परिणाम->op     = GGML_OP_GET_ROWS;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_get_rows_back

काष्ठा ggml_tensor * ggml_get_rows_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1]);

    परिणाम->op     = GGML_OP_GET_ROWS_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_set_rows

काष्ठा ggml_tensor * ggml_set_rows(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c) {
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

    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SET_ROWS;
    परिणाम->src[0] = b;
    परिणाम->src[1] = c;
    परिणाम->src[2] = a; // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

    वापस परिणाम;
}

// ggml_diag

काष्ठा ggml_tensor * ggml_diag(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    GGML_ASSERT(a->ne[1] == 1);

    स्थिर int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, 4, ne);

    परिणाम->op     = GGML_OP_DIAG;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_diag_mask_inf

अटल काष्ठा ggml_tensor * ggml_diag_mask_inf_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_DIAG_MASK_INF;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_diag_mask_inf(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past) {
    वापस ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

काष्ठा ggml_tensor * ggml_diag_mask_inf_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past) {
    वापस ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// ggml_diag_mask_zero

अटल काष्ठा ggml_tensor * ggml_diag_mask_zero_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_DIAG_MASK_ZERO;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_diag_mask_zero(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past) {
    वापस ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

काष्ठा ggml_tensor * ggml_diag_mask_zero_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_past) {
    वापस ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// ggml_clamp

अटल काष्ठा ggml_tensor * ggml_clamp_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        भग्न                 min,
        भग्न                 max,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भग्न params[] = { min, max };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_CLAMP;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_clamp(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    भग्न                 min,
    भग्न                 max) {
    वापस ggml_clamp_impl(ctx, a, min, max, false);
}

काष्ठा ggml_tensor * ggml_clamp_inplace(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    भग्न                 min,
    भग्न                 max) {
    वापस ggml_clamp_impl(ctx, a, min, max, true);
}

// ggml_soft_max

अटल काष्ठा ggml_tensor * ggml_soft_max_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * mask,
        भग्न                 scale,
        भग्न                 max_bias,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous(a));

    अगर (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
        GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);
        GGML_ASSERT(a->ne[3]%mask->ne[3] == 0);
    }

    अगर (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    भग्न params[] = { scale, max_bias };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_SOFT_MAX;
    परिणाम->src[0] = a;
    परिणाम->src[1] = mask;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_soft_max(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_soft_max_impl(ctx, a, शून्य, 1.0f, 0.0f, false);
}

काष्ठा ggml_tensor * ggml_soft_max_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a) {
    वापस ggml_soft_max_impl(ctx, a, शून्य, 1.0f, 0.0f, true);
}

काष्ठा ggml_tensor * ggml_soft_max_ext(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * mask,
        भग्न                 scale,
        भग्न                 max_bias) {
    वापस ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

काष्ठा ggml_tensor * ggml_soft_max_ext_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * mask,
        भग्न                 scale,
        भग्न                 max_bias) {
    वापस ggml_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

व्योम ggml_soft_max_add_sinks(
        काष्ठा ggml_tensor * a,
        काष्ठा ggml_tensor * sinks) {
    अगर (!sinks) {
        a->src[2] = शून्य;
        वापस;
    }

    GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
    GGML_ASSERT(a->src[2] == शून्य);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[2] = sinks;
}

// ggml_soft_max_ext_back

अटल काष्ठा ggml_tensor * ggml_soft_max_ext_back_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 scale,
        भग्न                 max_bias,
        bool                  inplace) {
    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    परिणाम->op     = GGML_OP_SOFT_MAX_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    स_नकल((भग्न *) परिणाम->op_params + 0, &scale,    माप(भग्न));
    स_नकल((भग्न *) परिणाम->op_params + 1, &max_bias, माप(भग्न));

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_soft_max_ext_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 scale,
        भग्न                 max_bias) {
    वापस ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

काष्ठा ggml_tensor * ggml_soft_max_ext_back_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        भग्न                 scale,
        भग्न                 max_bias) {
    वापस ggml_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// ggml_rope

अटल काष्ठा ggml_tensor * ggml_rope_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   sections[GGML_MROPE_SECTIONS],
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow,
        bool                  inplace) {
    GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    GGML_ASSERT(ggml_is_vector(b));
    GGML_ASSERT(b->type == GGML_TYPE_I32);

    bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;
    अगर (mrope_used) {
        GGML_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
    } अन्यथा {
        GGML_ASSERT(a->ne[2] == b->ne[0]);
    }

    अगर (c) {
        GGML_ASSERT(c->type == GGML_TYPE_F32);
        GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    int32_t params[16] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    स_नकल(params +  5, &freq_base,    माप(भग्न));
    स_नकल(params +  6, &freq_scale,   माप(भग्न));
    स_नकल(params +  7, &ext_factor,   माप(भग्न));
    स_नकल(params +  8, &attn_factor,  माप(भग्न));
    स_नकल(params +  9, &beta_fast,    माप(भग्न));
    स_नकल(params + 10, &beta_slow,    माप(भग्न));
    अगर (mrope_used && sections) {
        स_नकल(params + 11, sections,  माप(int32_t) * GGML_MROPE_SECTIONS);
    } अन्यथा {
        स_रखो(params + 11, 0,         माप(int32_t) * GGML_MROPE_SECTIONS);
    }
    params[15] = 0; // n_offs, set via ggml_rope_set_offset()

    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_ROPE;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_rope(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode) {
    वापस ggml_rope_impl(
        ctx, a, b, शून्य, n_dims, शून्य, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

काष्ठा ggml_tensor * ggml_rope_multi(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   sections[GGML_MROPE_SECTIONS],
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

काष्ठा ggml_tensor * ggml_rope_multi_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   sections[GGML_MROPE_SECTIONS],
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

काष्ठा ggml_tensor * ggml_rope_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode) {
    वापस ggml_rope_impl(
        ctx, a, b, शून्य, n_dims, शून्य, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

काष्ठा ggml_tensor * ggml_rope_ext(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, c, n_dims, शून्य, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

काष्ठा ggml_tensor * ggml_rope_ext_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, c, n_dims, शून्य, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

काष्ठा ggml_tensor * ggml_rope_custom(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, शून्य, n_dims, शून्य, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

काष्ठा ggml_tensor * ggml_rope_custom_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    वापस ggml_rope_impl(
        ctx, a, b, शून्य, n_dims, शून्य, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
अटल भग्न ggml_rope_yarn_corr_dim(पूर्णांक n_dims, पूर्णांक n_ctx_orig, भग्न n_rot, भग्न base) {
    वापस n_dims * logf(n_ctx_orig / (n_rot * 2 * (भग्न)M_PI)) / (2 * logf(base));
}

व्योम ggml_rope_yarn_corr_dims(
    पूर्णांक n_dims, पूर्णांक n_ctx_orig, भग्न freq_base, भग्न beta_fast, भग्न beta_slow, भग्न dims[2]
) {
    // start and end correction dims
    भग्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    भग्न end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// ggml_rope_back

काष्ठा ggml_tensor * ggml_rope_ext_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    काष्ठा ggml_tensor * परिणाम = ggml_rope_ext(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    परिणाम->op = GGML_OP_ROPE_BACK;
    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_rope_multi_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c,
        पूर्णांक                   n_dims,
        पूर्णांक                   sections[4],
        पूर्णांक                   mode,
        पूर्णांक                   n_ctx_orig,
        भग्न                 freq_base,
        भग्न                 freq_scale,
        भग्न                 ext_factor,
        भग्न                 attn_factor,
        भग्न                 beta_fast,
        भग्न                 beta_slow) {
    काष्ठा ggml_tensor * परिणाम = ggml_rope_multi(
        ctx, a, b, c, n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    परिणाम->op = GGML_OP_ROPE_BACK;
    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_rope_set_offset(
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   n_offs) {
    GGML_ASSERT(a->op == GGML_OP_ROPE || a->op == GGML_OP_ROPE_BACK);
    GGML_ASSERT(n_offs >= 0);

    स्थिर int32_t mode = ggml_get_op_params_i32(a, 2);
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    ggml_set_op_params_i32(a, 15, n_offs);
    वापस a;
}

अटल int64_t ggml_calc_conv_output_आकार(int64_t ins, int64_t ks, पूर्णांक s, पूर्णांक p, पूर्णांक d) {
    वापस (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
काष्ठा ggml_tensor * ggml_im2col(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   d0,
        पूर्णांक                   d1,
        bool                  is_2D,
        क्रमागत ggml_type        dst_type) {
    अगर (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } अन्यथा {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    स्थिर int64_t OH = is_2D ? ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    स्थिर int64_t OW =         ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    स्थिर int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_IM2COL;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_im2col_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        int64_t             * ne,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   d0,
        पूर्णांक                   d1,
        bool                  is_2D) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_IM2COL_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_conv_1d

काष्ठा ggml_tensor * ggml_conv_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   p0,
        पूर्णांक                   d0) {
    काष्ठा ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OL, IC * K]

    काष्ठा ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    परिणाम = ggml_reshape_3d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], a->ne[2]); // [OC, N, OL]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 2, 1, 3));                // [N, OC, OL]

    वापस परिणाम;
}

// ggml_conv_1d_ph

काष्ठा ggml_tensor* ggml_conv_1d_ph(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s,
        पूर्णांक                   d) {
    वापस ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// ggml_conv_1d_dw

काष्ठा ggml_tensor * ggml_conv_1d_dw(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   p0,
        पूर्णांक                   d0) {
    काष्ठा ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

    काष्ठा ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16);

    काष्ठा ggml_tensor * परिणाम = ggml_mul_mat(ctx, im2col, a);

    परिणाम = ggml_reshape_3d(ctx, परिणाम, परिणाम->ne[0], परिणाम->ne[2], 1);

    वापस परिणाम;
}

// ggml_conv_1d_dw_ph

काष्ठा ggml_tensor * ggml_conv_1d_dw_ph(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   d0) {
    वापस ggml_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// ggml_col2im_1d

काष्ठा ggml_tensor * ggml_col2im_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   s0,
        पूर्णांक                   oc,
        पूर्णांक                   p0) {
    GGML_ASSERT(ggml_is_matrix(a));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(oc > 0);
    GGML_ASSERT(p0 >= 0);

    स्थिर int64_t K_OC = a->ne[0];
    स्थिर int64_t T_in = a->ne[1];
    स्थिर int64_t K = K_OC / oc;
    स्थिर int64_t T_out = (T_in - 1) * s0 + K - 2 * p0;

    GGML_ASSERT(K_OC == K * oc);  // a->ne[0] must be a whole number of oc blocks
    GGML_ASSERT(K > 0 && T_out > 0);

    स्थिर int64_t ne[4] = { T_out, oc, 1, 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, a->type, 2, ne);

    int32_t params[] = { s0, (int32_t)oc, (int32_t)p0 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_COL2IM_1D;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_conv_transpose_1d

अटल int64_t ggml_calc_conv_transpose_1d_output_आकार(int64_t ins, int64_t ks, पूर्णांक s, पूर्णांक p, पूर्णांक d) {
    वापस (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

GGML_API काष्ठा ggml_tensor * ggml_conv_transpose_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   p0,
        पूर्णांक                   d0) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[2] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);

    GGML_ASSERT(p0 == 0);
    GGML_ASSERT(d0 == 1);

    स्थिर int64_t ne[4] = {
        ggml_calc_conv_transpose_1d_output_आकार(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_CONV_TRANSPOSE_1D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
काष्ठा ggml_tensor * ggml_conv_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   d0,
        पूर्णांक                   d1) {
    काष्ठा ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

    काष्ठा ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 1, 3, 2)); // [N, OC, OH, OW]


    वापस परिणाम;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
काष्ठा ggml_tensor * ggml_im2col_3d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        int64_t               IC,
        पूर्णांक                   s0, // stride width
        पूर्णांक                   s1, // stride height
        पूर्णांक                   s2, // stride depth
        पूर्णांक                   p0, // padding width
        पूर्णांक                   p1, // padding height
        पूर्णांक                   p2, // padding depth
        पूर्णांक                   d0, // dilation width
        पूर्णांक                   d1, // dilation height
        पूर्णांक                   d2, // dilation depth
        क्रमागत ggml_type        dst_type) {
    स्थिर int64_t N = b->ne[3] / IC;
    स्थिर int64_t ID = b->ne[2];
    स्थिर int64_t IH = b->ne[1];
    स्थिर int64_t IW = b->ne[0];

    स्थिर int64_t OC = a->ne[3] / IC;
    UNUSED(OC);
    स्थिर int64_t KD = a->ne[2];
    स्थिर int64_t KH = a->ne[1];
    स्थिर int64_t KW = a->ne[0];
    स्थिर int64_t OD = ggml_calc_conv_output_आकार(ID, KD, s2, p2, d2);
    स्थिर int64_t OH = ggml_calc_conv_output_आकार(IH, KH, s1, p1, d1);
    स्थिर int64_t OW = ggml_calc_conv_output_आकार(IW, KW, s0, p0, d0);

    GGML_ASSERT((OD > 0)  && "b too small compared to a");
    GGML_ASSERT((OH > 0)  && "b too small compared to a");
    GGML_ASSERT((OW > 0)  && "b too small compared to a");


    स्थिर int64_t ne[4] = {KW*KH*KD*IC, OW, OH, OD*N};

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_IM2COL_3D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
काष्ठा ggml_tensor * ggml_conv_3d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        int64_t               IC,
        पूर्णांक                   s0, // stride width
        पूर्णांक                   s1, // stride height
        पूर्णांक                   s2, // stride depth
        पूर्णांक                   p0, // padding width
        पूर्णांक                   p1, // padding height
        पूर्णांक                   p2, // padding depth
        पूर्णांक                   d0, // dilation width
        पूर्णांक                   d1, // dilation height
        पूर्णांक                   d2  // dilation depth
        ) {
    काष्ठा ggml_tensor * im2col = ggml_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N*OD, OH, OW, IC * KD * KH * KW]

    int64_t OC = a->ne[3] / IC;
    int64_t N = b->ne[3] / IC;
    काष्ठा ggml_tensor * परिणाम =
        ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
                ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));                          // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

    int64_t OD = im2col->ne[3] / N;
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1]*im2col->ne[2], OD, N, OC); // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
    परिणाम = ggml_cont(ctx, ggml_permute(ctx, परिणाम, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

    वापस परिणाम;
}

// ggml_conv_2d_sk_p0

काष्ठा ggml_tensor * ggml_conv_2d_sk_p0(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// ggml_conv_2d_s1_ph

काष्ठा ggml_tensor * ggml_conv_2d_s1_ph(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    वापस ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// ggml_conv_2d_dw

काष्ठा ggml_tensor * ggml_conv_2d_dw(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   d0,
        पूर्णांक                   d1) {
    काष्ठा ggml_tensor * new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    काष्ठा ggml_tensor * im2col = ggml_im2col(ctx, new_a,
                                        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, a->type == GGML_TYPE_BF16 ? GGML_TYPE_F32 : GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    काष्ठा ggml_tensor * new_b = ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    काष्ठा ggml_tensor * परिणाम = ggml_mul_mat(ctx, new_a, new_b);
    परिणाम = ggml_reshape_4d(ctx, परिणाम, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    वापस परिणाम;
}

// ggml_conv_2d_dw_direct

काष्ठा ggml_tensor * ggml_conv_2d_dw_direct(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   stride0,
        पूर्णांक                   stride1,
        पूर्णांक                   pad0,
        पूर्णांक                   pad1,
        पूर्णांक                   dilation0,
        पूर्णांक                   dilation1) {
    GGML_ASSERT(a->ne[2] == 1);
    GGML_ASSERT(a->ne[3] == b->ne[2]);
    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], stride0, pad0, dilation0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], stride1, pad1, dilation1);
    ne[2] = b->ne[2];
    ne[3] = b->ne[3];

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, b->type, 4, ne);

    अगर (ggml_is_contiguous_channels(b)) {
        // Result will be permuted the same way as input (CWHN order)
        स्थिर int64_t type_आकार = ggml_type_आकार(परिणाम->type);
        GGML_ASSERT(ggml_blck_आकार(परिणाम->type) == 1);
        परिणाम->nb[0] = परिणाम->ne[2] * type_आकार;
        परिणाम->nb[1] = परिणाम->ne[0] * परिणाम->nb[0];
        परिणाम->nb[2] = type_आकार;
    }

    int32_t params[] = { stride0, stride1, pad0, pad1, dilation0, dilation1 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_CONV_2D_DW;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    वापस परिणाम;
}

// ggml_conv_2d_direct

काष्ठा ggml_tensor * ggml_conv_2d_direct(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,   // convolution kernel [KW, KH, IC, OC]
        काष्ठा ggml_tensor  * b,   // input data [W, H, C, N]
        पूर्णांक                   s0,  // stride dimension 0
        पूर्णांक                   s1,  // stride dimension 1
        पूर्णांक                   p0,  // padding dimension 0
        पूर्णांक                   p1,  // padding dimension 1
        पूर्णांक                   d0,  // dilation dimension 0
        पूर्णांक                   d1) {// dilation dimension 1

    GGML_ASSERT(a->ne[2] == b->ne[2]);
    //GGML_ASSERT(a->type == b->type);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = a->ne[3];
    ne[3] = b->ne[3];

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, b->type, 4, ne);

    ggml_set_op_params_i32(परिणाम, 0, s0);
    ggml_set_op_params_i32(परिणाम, 1, s1);
    ggml_set_op_params_i32(परिणाम, 2, p0);
    ggml_set_op_params_i32(परिणाम, 3, p1);
    ggml_set_op_params_i32(परिणाम, 4, d0);
    ggml_set_op_params_i32(परिणाम, 5, d1);

    परिणाम->op = GGML_OP_CONV_2D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_conv_3d_direct

काष्ठा ggml_tensor * ggml_conv_3d_direct(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        पूर्णांक                   s2,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   p2,
        पूर्णांक                   d0,
        पूर्णांक                   d1,
        पूर्णांक                   d2,
        पूर्णांक                   c,
        पूर्णांक                   n,
        पूर्णांक                   oc) {

    GGML_ASSERT(a->ne[3] == (int64_t) c * oc);
    GGML_ASSERT(b->ne[3] == (int64_t) c * n);

    int64_t ne[4];
    ne[0] = ggml_calc_conv_output_आकार(b->ne[0], a->ne[0], s0, p0, d0);
    ne[1] = ggml_calc_conv_output_आकार(b->ne[1], a->ne[1], s1, p1, d1);
    ne[2] = ggml_calc_conv_output_आकार(b->ne[2], a->ne[2], s2, p2, d2);
    ne[3] = (int64_t) oc * n;

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(परिणाम, 0,  s0);
    ggml_set_op_params_i32(परिणाम, 1,  s1);
    ggml_set_op_params_i32(परिणाम, 2,  s2);
    ggml_set_op_params_i32(परिणाम, 3,  p0);
    ggml_set_op_params_i32(परिणाम, 4,  p1);
    ggml_set_op_params_i32(परिणाम, 5,  p2);
    ggml_set_op_params_i32(परिणाम, 6,  d0);
    ggml_set_op_params_i32(परिणाम, 7,  d1);
    ggml_set_op_params_i32(परिणाम, 8,  d2);
    ggml_set_op_params_i32(परिणाम, 9,  c);
    ggml_set_op_params_i32(परिणाम, 10, n);
    ggml_set_op_params_i32(परिणाम, 11, oc);

    परिणाम->op = GGML_OP_CONV_3D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_conv_transpose_2d_p0

अटल int64_t ggml_calc_conv_transpose_output_आकार(int64_t ins, int64_t ks, पूर्णांक s, पूर्णांक p) {
    वापस (ins - 1) * s - 2 * p + ks;
}

काष्ठा ggml_tensor * ggml_conv_transpose_2d_p0(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        पूर्णांक                   stride) {
    GGML_ASSERT(a->ne[3] == b->ne[2]);

    स्थिर int64_t ne[4] = {
        ggml_calc_conv_transpose_output_आकार(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        ggml_calc_conv_transpose_output_आकार(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    काष्ठा ggml_tensor* परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(परिणाम, 0, stride);

    परिणाम->op     = GGML_OP_CONV_TRANSPOSE_2D;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_pool_*

अटल int64_t ggml_calc_pool_output_आकार(int64_t ins, पूर्णांक ks, पूर्णांक s, भग्न p) {
    वापस (ins + 2 * p - ks) / s + 1;
}

// ggml_pool_1d

काष्ठा ggml_tensor * ggml_pool_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_op_pool     op,
        पूर्णांक                   k0,
        पूर्णांक                   s0,
        पूर्णांक                   p0) {
    स्थिर int64_t ne[4] = {
        ggml_calc_pool_output_आकार(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_POOL_1D;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_pool_2d

काष्ठा ggml_tensor * ggml_pool_2d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_op_pool     op,
        पूर्णांक                   k0,
        पूर्णांक                   k1,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        भग्न                 p0,
        भग्न                 p1) {
    काष्ठा ggml_tensor * परिणाम;
    स्थिर int64_t ne[4] = {
        ggml_calc_pool_output_आकार(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_आकार(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    GGML_ASSERT(ne[0] > 0);
    GGML_ASSERT(ne[1] > 0);

    परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_POOL_2D;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_pool_2d_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * af,
        क्रमागत ggml_op_pool     op,
        पूर्णांक                   k0,
        पूर्णांक                   k1,
        पूर्णांक                   s0,
        पूर्णांक                   s1,
        भग्न                 p0,
        भग्न                 p1) {
    काष्ठा ggml_tensor * परिणाम;
    परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_POOL_2D_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = af;

    वापस परिणाम;
}

// ggml_upscale / ggml_interpolate

अटल काष्ठा ggml_tensor * ggml_interpolate_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    GGML_ASSERT((mode & 0xFF) < GGML_SCALE_MODE_गिनती);
    // TODO: implement antialias for modes other than bilinear
    GGML_ASSERT(!(mode & GGML_SCALE_FLAG_ANTIALIAS) || (mode & 0xFF) == GGML_SCALE_MODE_BILINEAR);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t)mode);

    परिणाम->op     = GGML_OP_UPSCALE;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_upscale(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   scale_factor,
        क्रमागत ggml_scale_mode  mode) {
    GGML_ASSERT(scale_factor > 1);
    वापस ggml_interpolate_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3], mode);
}

काष्ठा ggml_tensor * ggml_upscale_ext(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   ne0,
        पूर्णांक                   ne1,
        पूर्णांक                   ne2,
        पूर्णांक                   ne3,
        क्रमागत ggml_scale_mode  mode) {
    वापस ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

काष्ठा ggml_tensor * ggml_interpolate(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        uint32_t              mode) {
    वापस ggml_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

// ggml_pad

काष्ठा ggml_tensor * ggml_pad(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   p2,
        पूर्णांक                   p3) {
    वापस ggml_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

// ggml_pad_circular

काष्ठा ggml_tensor * ggml_pad_circular(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   p0,
        पूर्णांक                   p1,
        पूर्णांक                   p2,
        पूर्णांक                   p3) {
    वापस ggml_pad_ext_circular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

काष्ठा ggml_tensor * ggml_pad_ext(
            काष्ठा ggml_context * ctx,
            काष्ठा ggml_tensor  * a,
            पूर्णांक                  lp0,
            पूर्णांक                  rp0,
            पूर्णांक                  lp1,
            पूर्णांक                  rp1,
            पूर्णांक                  lp2,
            पूर्णांक                  rp2,
            पूर्णांक                  lp3,
            पूर्णांक                  rp3
            ) {
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + lp0 + rp0,
            a->ne[1] + lp1 + rp1,
            a->ne[2] + lp2 + rp2,
            a->ne[3] + lp3 + rp3);

    ggml_set_op_params_i32(परिणाम, 0, lp0);
    ggml_set_op_params_i32(परिणाम, 1, rp0);
    ggml_set_op_params_i32(परिणाम, 2, lp1);
    ggml_set_op_params_i32(परिणाम, 3, rp1);
    ggml_set_op_params_i32(परिणाम, 4, lp2);
    ggml_set_op_params_i32(परिणाम, 5, rp2);
    ggml_set_op_params_i32(परिणाम, 6, lp3);
    ggml_set_op_params_i32(परिणाम, 7, rp3);
    ggml_set_op_params_i32(परिणाम, 8, 0); // not circular by default


    परिणाम->op     = GGML_OP_PAD;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_pad_ext_circular

काष्ठा ggml_tensor * ggml_pad_ext_circular(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                  lp0,
        पूर्णांक                  rp0,
        पूर्णांक                  lp1,
        पूर्णांक                  rp1,
        पूर्णांक                  lp2,
        पूर्णांक                  rp2,
        पूर्णांक                  lp3,
        पूर्णांक                  rp3
        ) {
    काष्ठा ggml_tensor * परिणाम = ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
    ggml_set_op_params_i32(परिणाम, 8, 1); // circular
    वापस परिणाम;
}

// ggml_pad_reflect_1d

काष्ठा ggml_tensor * ggml_pad_reflect_1d(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   p0,
        पूर्णांक                   p1) {
    GGML_ASSERT(p0 >= 0);
    GGML_ASSERT(p1 >= 0);

    GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
    GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0 + p1,
            a->ne[1],
            a->ne[2],
            a->ne[3]);

    int32_t params[] = { p0, p1 };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_PAD_REFLECT_1D;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_roll

काष्ठा ggml_tensor * ggml_roll(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   shift0,
        पूर्णांक                   shift1,
        पूर्णांक                   shift2,
        पूर्णांक                   shift3) {
    GGML_ASSERT(a->nb[0] == ggml_type_आकार(a->type));
    GGML_ASSERT(असल(shift0) < a->ne[0]);
    GGML_ASSERT(असल(shift1) < a->ne[1]);
    GGML_ASSERT(असल(shift2) < a->ne[2]);
    GGML_ASSERT(असल(shift3) < a->ne[3]);

    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(परिणाम, 0, shift0);
    ggml_set_op_params_i32(परिणाम, 1, shift1);
    ggml_set_op_params_i32(परिणाम, 2, shift2);
    ggml_set_op_params_i32(परिणाम, 3, shift3);

    परिणाम->op     = GGML_OP_ROLL;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_timestep_embedding

काष्ठा ggml_tensor * ggml_timestep_embedding(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * timesteps,
        पूर्णांक                   dim,
        पूर्णांक                   max_period) {

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, timesteps->ne[0]);

    ggml_set_op_params_i32(परिणाम, 0, dim);
    ggml_set_op_params_i32(परिणाम, 1, max_period);

    परिणाम->op     = GGML_OP_TIMESTEP_EMBEDDING;
    परिणाम->src[0] = timesteps;

    वापस परिणाम;
}

// ggml_tri

काष्ठा ggml_tensor * ggml_tri(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    क्रमागत ggml_tri_type    type) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(a->ne[0] == a->ne[1]);

    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(परिणाम, 0, type);

    परिणाम->op = GGML_OP_TRI;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_fill

अटल काष्ठा ggml_tensor * ggml_fill_impl(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    भग्न                 c,
    bool                  inplace) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_f32(परिणाम, 0, c);

    परिणाम->op = GGML_OP_FILL;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_fill(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    भग्न                 c) {
    वापस ggml_fill_impl(ctx, a, c, false);
}

काष्ठा ggml_tensor * ggml_fill_inplace(
    काष्ठा ggml_context * ctx,
    काष्ठा ggml_tensor  * a,
    भग्न                 c) {
    वापस ggml_fill_impl(ctx, a, c, true);
}

// ggml_argsort

काष्ठा ggml_tensor * ggml_argsort(
        काष्ठा ggml_context  * ctx,
        काष्ठा ggml_tensor   * a,
        क्रमागत ggml_sort_order   order) {
    GGML_ASSERT(a->ne[0] <= INT32_MAX);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) order);

    परिणाम->op     = GGML_OP_ARGSORT;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_argsort_top_k

काष्ठा ggml_tensor * ggml_argsort_शीर्ष_k(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    काष्ठा ggml_tensor * परिणाम = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    परिणाम = ggml_view_4d(ctx, परिणाम,
                k, परिणाम->ne[1], परिणाम->ne[2], परिणाम->ne[3],
                   परिणाम->nb[1], परिणाम->nb[2], परिणाम->nb[3],
                0);

    वापस परिणाम;
}

// ggml_top_k

काष्ठा ggml_tensor * ggml_शीर्ष_k(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, a->ne[1], a->ne[2], a->ne[3]);

    परिणाम->op     = GGML_OP_शीर्ष_K;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_arange

काष्ठा ggml_tensor * ggml_arange(
        काष्ठा ggml_context * ctx,
        भग्न                 start,
        भग्न                 stop,
        भग्न                 step) {
    GGML_ASSERT(stop > start);

    स्थिर int64_t steps = (int64_t) ceilf((stop - start) / step);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, steps);

    ggml_set_op_params_f32(परिणाम, 0, start);
    ggml_set_op_params_f32(परिणाम, 1, stop);
    ggml_set_op_params_f32(परिणाम, 2, step);

    परिणाम->op = GGML_OP_ARANGE;

    वापस परिणाम;
}

// ggml_flash_attn_ext

काष्ठा ggml_tensor * ggml_flash_attn_ext(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * q,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * mask,
        भग्न                 scale,
        भग्न                 max_bias,
        भग्न                 logit_softcap) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    GGML_ASSERT(q->ne[3] == k->ne[3]);
    GGML_ASSERT(q->ne[3] == v->ne[3]);

    अगर (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(mask));
        //GGML_ASSERT(ggml_can_repeat_rows(mask, qk));

        GGML_ASSERT(q->ne[2] % mask->ne[2] == 0);
        GGML_ASSERT(q->ne[3] % mask->ne[3] == 0);
    }

    अगर (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    भग्न params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_FLASH_ATTN_EXT;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = mask;

    वापस परिणाम;
}


व्योम ggml_flash_attn_ext_set_prec(
        काष्ठा ggml_tensor * a,
        क्रमागत ggml_prec       prec) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    स्थिर int32_t prec_i32 = (int32_t) prec;

    ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

क्रमागत ggml_prec ggml_flash_attn_ext_get_prec(
        स्थिर काष्ठा ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

    स्थिर int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

    वापस (क्रमागत ggml_prec) prec_i32;
}

व्योम ggml_flash_attn_ext_set_n_kv_max(
        काष्ठा ggml_tensor * a,
        int32_t              n_kv_max) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(n_kv_max >= 0);

    ggml_set_op_params_i32(a, 4, n_kv_max);
}

व्योम ggml_flash_attn_ext_add_sinks(
        काष्ठा ggml_tensor * a,
        काष्ठा ggml_tensor * sinks) {
    अगर (!sinks) {
        a->src[4] = शून्य;
        वापस;
    }

    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[4] == शून्य);
    GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
    GGML_ASSERT(sinks->type == GGML_TYPE_F32);

    a->src[4] = sinks;
}

// ggml_flash_attn_back

काष्ठा ggml_tensor * ggml_flash_attn_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * q,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * d,
        bool                  masked) {
    GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes");

    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    स्थिर int64_t     D = q->ne[0];
    स्थिर int64_t     N = q->ne[1];
    स्थिर int64_t     M = k->ne[1];
    स्थिर int64_t   ne2 = q->ne[2];
    स्थिर int64_t   ne3 = q->ne[3];
    स्थिर int64_t kvne2 = k->ne[2];

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
    स्थिर int64_t elem_q = ggml_nelements(q);
    स्थिर int64_t elem_k = ggml_nelements(k);
    स्थिर int64_t elem_v = ggml_nelements(v);

    क्रमागत ggml_type परिणाम_type = GGML_TYPE_F32;
    GGML_ASSERT(ggml_blck_आकार(परिणाम_type) == 1);
    स्थिर माप_प्रकार tsize = ggml_type_आकार(परिणाम_type);

    स्थिर माप_प्रकार offs_q = 0;
    स्थिर माप_प्रकार offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    स्थिर माप_प्रकार offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);
    स्थिर माप_प्रकार end    = offs_v + GGML_PAD(elem_v * tsize, GGML_MEM_ALIGN);

    स्थिर माप_प्रकार nelements = (end + tsize - 1)/tsize;

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    ggml_set_op_params(परिणाम, &masked_i, माप(masked_i));

    परिणाम->op     = GGML_OP_FLASH_ATTN_BACK;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = d;

    वापस परिणाम;
}

// ggml_ssm_conv

काष्ठा ggml_tensor * ggml_ssm_conv(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * sx,
        काष्ठा ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_3d(sx));
    GGML_ASSERT(ggml_is_matrix(c));

    स्थिर int64_t d_conv  = c->ne[0];
    स्थिर int64_t d_inner = c->ne[1];
    स्थिर int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    स्थिर int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(n_t >= 0);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);

    परिणाम->op     = GGML_OP_SSM_CONV;
    परिणाम->src[0] = sx;
    परिणाम->src[1] = c;

    वापस परिणाम;
}

// ggml_ssm_scan

काष्ठा ggml_tensor * ggml_ssm_scan(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * s,
        काष्ठा ggml_tensor  * x,
        काष्ठा ggml_tensor  * dt,
        काष्ठा ggml_tensor  * A,
        काष्ठा ggml_tensor  * B,
        काष्ठा ggml_tensor  * C,
        काष्ठा ggml_tensor  * ids,
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
        स्थिर int64_t d_state      = s->ne[0];
        स्थिर int64_t शीर्ष_dim     = x->ne[0];
        स्थिर int64_t n_शीर्ष       = x->ne[1];
        स्थिर int64_t n_seq_tokens = x->ne[2];
        स्थिर int64_t n_seqs       = x->ne[3];

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
        GGML_ASSERT(ggml_is_vector(ids));
        GGML_ASSERT(A->ne[1] == n_शीर्ष);
        GGML_ASSERT(ggml_is_matrix(A));

        अगर (A->ne[0] != 1) {
            // Mamba-1 has more granular decay factors
            GGML_ASSERT(A->ne[0] == d_state);
            GGML_ASSERT(K == 1);
        }
    }

    // concatenated y + ssm_states
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + K*s->ne[0]*s->ne[1]*s->ne[2]*ids->ne[0]);

    परिणाम->op   = GGML_OP_SSM_SCAN;
    परिणाम->src[0] = s;
    परिणाम->src[1] = x;
    परिणाम->src[2] = dt;
    परिणाम->src[3] = A;
    परिणाम->src[4] = B;
    परिणाम->src[5] = C;
    परिणाम->src[6] = ids;

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) K);

    वापस परिणाम;
}

// ggml_win_part

काष्ठा ggml_tensor * ggml_win_part(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   w) {
    GGML_ASSERT(a->ne[3] == 1);
    GGML_ASSERT(a->type  == GGML_TYPE_F32);

    // padding
    स्थिर पूर्णांक px = (w - a->ne[1]%w)%w;
    स्थिर पूर्णांक py = (w - a->ne[2]%w)%w;

    स्थिर पूर्णांक npx = (px + a->ne[1])/w;
    स्थिर पूर्णांक npy = (py + a->ne[2])/w;
    स्थिर पूर्णांक np  = npx*npy;

    स्थिर int64_t ne[4] = { a->ne[0], w, w, np, };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_WIN_PART;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_win_unpart

काष्ठा ggml_tensor * ggml_win_unpart(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   w0,
        पूर्णांक                   h0,
        पूर्णांक                   w) {
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    स्थिर int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    ggml_set_op_params(परिणाम, params, माप(params));

    परिणाम->op     = GGML_OP_WIN_UNPART;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_get_rel_pos

काष्ठा ggml_tensor * ggml_get_rel_pos(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        पूर्णांक                   qh,
        पूर्णांक                   kh) {
    GGML_ASSERT(qh == kh);
    GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    स्थिर int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, ne);

    परिणाम->op     = GGML_OP_GET_REL_POS;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

// ggml_add_rel_pos

अटल काष्ठा ggml_tensor * ggml_add_rel_pos_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * pw,
        काष्ठा ggml_tensor  * ph,
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

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    ggml_set_op_params_i32(परिणाम, 0, inplace ? 1 : 0);

    परिणाम->op     = GGML_OP_ADD_REL_POS;
    परिणाम->src[0] = a;
    परिणाम->src[1] = pw;
    परिणाम->src[2] = ph;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_add_rel_pos(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * pw,
        काष्ठा ggml_tensor  * ph) {
    वापस ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

काष्ठा ggml_tensor * ggml_add_rel_pos_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * pw,
        काष्ठा ggml_tensor  * ph) {
    वापस ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// ggml_rwkv_wkv6

काष्ठा ggml_tensor * ggml_rwkv_wkv6(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * r,
        काष्ठा ggml_tensor  * tf,
        काष्ठा ggml_tensor  * td,
        काष्ठा ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(tf));
    GGML_ASSERT(ggml_is_contiguous(td));
    GGML_ASSERT(ggml_is_contiguous(state));

    स्थिर int64_t S = k->ne[0];
    स्थिर int64_t H = k->ne[1];
    स्थिर int64_t n_tokens = k->ne[2];
    स्थिर int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
        GGML_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    स्थिर int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_RWKV_WKV6;
    परिणाम->src[0] = k;
    परिणाम->src[1] = v;
    परिणाम->src[2] = r;
    परिणाम->src[3] = tf;
    परिणाम->src[4] = td;
    परिणाम->src[5] = state;

    वापस परिणाम;
}

// ggml_gated_linear_attn

काष्ठा ggml_tensor * ggml_gated_linear_attn(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * q,
        काष्ठा ggml_tensor  * g,
        काष्ठा ggml_tensor  * state,
        भग्न scale) {
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(q));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(state));

    स्थिर int64_t S = k->ne[0];
    स्थिर int64_t H = k->ne[1];
    स्थिर int64_t n_tokens = k->ne[2];
    स्थिर int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
        GGML_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    स्थिर int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_f32(परिणाम, 0, scale);

    परिणाम->op     = GGML_OP_GATED_LINEAR_ATTN;
    परिणाम->src[0] = k;
    परिणाम->src[1] = v;
    परिणाम->src[2] = q;
    परिणाम->src[3] = g;
    परिणाम->src[4] = state;

    वापस परिणाम;
}

// ggml_rwkv_wkv7

काष्ठा ggml_tensor * ggml_rwkv_wkv7(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * r,
        काष्ठा ggml_tensor  * w,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * state) {
    GGML_ASSERT(ggml_is_contiguous(r));
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(k));
    GGML_ASSERT(ggml_is_contiguous(v));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_is_contiguous(state));

    स्थिर int64_t S = k->ne[0];
    स्थिर int64_t H = k->ne[1];
    स्थिर int64_t n_tokens = k->ne[2];
    स्थिर int64_t n_seqs = state->ne[1];
    {
        GGML_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
        GGML_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
        GGML_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
        GGML_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
        GGML_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
        GGML_ASSERT(ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    स्थिर int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op     = GGML_OP_RWKV_WKV7;
    परिणाम->src[0] = r;
    परिणाम->src[1] = w;
    परिणाम->src[2] = k;
    परिणाम->src[3] = v;
    परिणाम->src[4] = a;
    परिणाम->src[5] = b;
    परिणाम->src[6] = state;

    वापस परिणाम;
}

// ggml_unary

अटल काष्ठा ggml_tensor * ggml_unary_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_unary_op    op,
        bool                  inplace) {
    GGML_ASSERT(ggml_is_contiguous_rows(a));

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) op);

    परिणाम->op     = GGML_OP_UNARY;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_unary(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_unary_op    op) {
    वापस ggml_unary_impl(ctx, a, op, false);
}

काष्ठा ggml_tensor * ggml_unary_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        क्रमागत ggml_unary_op    op) {
    वापस ggml_unary_impl(ctx, a, op, true);
}

// ggml_map_custom1

अटल काष्ठा ggml_tensor * ggml_map_custom1_impl(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        स्थिर  ggml_custom1_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    काष्ठा ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM1;
    परिणाम->src[0] = a;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_map_custom1(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        स्थिर  ggml_custom1_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

काष्ठा ggml_tensor * ggml_map_custom1_inplace(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        स्थिर  ggml_custom1_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// ggml_map_custom2

अटल काष्ठा ggml_tensor * ggml_map_custom2_impl(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        स्थिर  ggml_custom2_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    काष्ठा ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM2;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_map_custom2(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        स्थिर  ggml_custom2_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

काष्ठा ggml_tensor * ggml_map_custom2_inplace(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        स्थिर  ggml_custom2_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// ggml_map_custom3

अटल काष्ठा ggml_tensor * ggml_map_custom3_impl(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        काष्ठा ggml_tensor       * c,
        स्थिर  ggml_custom3_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata,
        bool                       inplace) {
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);

    काष्ठा ggml_tensor * परिणाम = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    काष्ठा ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op     = GGML_OP_MAP_CUSTOM3;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_map_custom3(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        काष्ठा ggml_tensor       * c,
        स्थिर  ggml_custom3_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

काष्ठा ggml_tensor * ggml_map_custom3_inplace(
        काष्ठा ggml_context      * ctx,
        काष्ठा ggml_tensor       * a,
        काष्ठा ggml_tensor       * b,
        काष्ठा ggml_tensor       * c,
        स्थिर  ggml_custom3_op_t   fun,
        पूर्णांक                        n_tasks,
        व्योम                     * userdata) {
    वापस ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

काष्ठा ggml_tensor * ggml_custom_4d(
        काष्ठा ggml_context * ctx,
        क्रमागत ggml_type        type,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        काष्ठा ggml_tensor ** args,
        पूर्णांक                   n_args,
        ggml_custom_op_t      fun,
        पूर्णांक                   n_tasks,
        व्योम                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);

    काष्ठा ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op = GGML_OP_CUSTOM;
    क्रम (पूर्णांक i = 0; i < n_args; i++) {
        परिणाम->src[i] = args[i];
    }

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_custom_inplace(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor ** args,
        पूर्णांक                   n_args,
        ggml_custom_op_t      fun,
        पूर्णांक                   n_tasks,
        व्योम                * userdata) {

    GGML_ASSERT(n_args < GGML_MAX_SRC - 1);

    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    काष्ठा ggml_custom_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    ggml_set_op_params(परिणाम, &params, माप(params));

    परिणाम->op = GGML_OP_CUSTOM;
    परिणाम->src[0] = a;
    क्रम (पूर्णांक i = 0; i < n_args; i++) {
        परिणाम->src[i + 1] = args[i];
    }

    वापस परिणाम;
}
// ggml_cross_entropy_loss

काष्ठा ggml_tensor * ggml_cross_entropy_loss(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_1d(ctx, a->type, 1);

    परिणाम->op     = GGML_OP_CROSS_ENTROPY_LOSS;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_cross_entropy_loss_back

काष्ठा ggml_tensor * ggml_cross_entropy_loss_back(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        काष्ठा ggml_tensor  * c) {
    GGML_ASSERT(ggml_is_scalar(a));
    GGML_ASSERT(ggml_are_same_shape(b, c));

    काष्ठा ggml_tensor * परिणाम = ggml_dup_tensor(ctx, b);

    परिणाम->op     = GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;
    परिणाम->src[2] = c;

    वापस परिणाम;
}

// opt_step_adamw

काष्ठा ggml_tensor * ggml_opt_step_adamw(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * grad,
        काष्ठा ggml_tensor  * m,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * adamw_params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(ggml_are_same_shape(a, m));
    GGML_ASSERT(ggml_are_same_shape(a, v));
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_OPT_STEP_ADAMW;
    परिणाम->src[0] = a;
    परिणाम->src[1] = grad;
    परिणाम->src[2] = m;
    परिणाम->src[3] = v;
    परिणाम->src[4] = adamw_params;

    वापस परिणाम;
}

// opt_step_sgd

काष्ठा ggml_tensor * ggml_opt_step_sgd(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * grad,
        काष्ठा ggml_tensor  * params) {
    GGML_ASSERT(a->flags & GGML_TENSOR_FLAG_PARAM);
    GGML_ASSERT(ggml_are_same_shape(a, grad));
    GGML_ASSERT(params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(params) == 2);

    काष्ठा ggml_tensor * परिणाम = ggml_view_tensor(ctx, a);

    परिणाम->op     = GGML_OP_OPT_STEP_SGD;
    परिणाम->src[0] = a;
    परिणाम->src[1] = grad;
    परिणाम->src[2] = params;

    वापस परिणाम;
}

// solve_tri

काष्ठा ggml_tensor * ggml_solve_tri(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * a,
        काष्ठा ggml_tensor  * b,
        bool                  बायाँ,
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

    GGML_ASSERT(lower && बायाँ && !uni); // TODO: support other variants

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b->ne[0], b->ne[1], b->ne[2], b->ne[3]);

    परिणाम->op     = GGML_OP_SOLVE_TRI;
    परिणाम->src[0] = a;
    परिणाम->src[1] = b;

    वापस परिणाम;
}

// ggml_gated_delta_net

काष्ठा ggml_tensor * ggml_gated_delta_net(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * q,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * v,
        काष्ठा ggml_tensor  * g,
        काष्ठा ggml_tensor  * beta,
        काष्ठा ggml_tensor  * state,
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

    स्थिर int64_t S_v      = v->ne[0];
    स्थिर int64_t H        = v->ne[1];
    स्थिर int64_t n_tokens = v->ne[2];
    स्थिर int64_t n_seqs   = v->ne[3];

    // gate: scalar [1, H, T, B] or vector [S_v, H, T, B] (KDA)
    GGML_ASSERT(g->ne[0] == 1 || g->ne[0] == S_v);
    GGML_ASSERT(beta->ne[0] == 1);

    // state holds the initial state s0 only: [S_v, S_v, H, n_seqs]. K (snapshot slot count) is an op param.
    GGML_ASSERT(state->ne[0] == S_v);
    GGML_ASSERT(state->ne[1] == S_v);
    GGML_ASSERT(state->ne[2] == H);
    GGML_ASSERT(state->ne[3] == n_seqs);
    GGML_ASSERT(K >= 1);
    स्थिर int64_t state_rows = K * S_v * n_seqs;
    स्थिर int64_t ne[4] = { S_v * H, n_tokens * n_seqs + state_rows, 1, 1 };
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    ggml_set_op_params_i32(परिणाम, 0, (int32_t) K);

    परिणाम->op     = GGML_OP_GATED_DELTA_NET;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = v;
    परिणाम->src[3] = g;
    परिणाम->src[4] = beta;
    परिणाम->src[5] = state;

    वापस परिणाम;
}

// ggml_lightning_indexer

काष्ठा ggml_tensor * ggml_lightning_indexer(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * q,
        काष्ठा ggml_tensor  * k,
        काष्ठा ggml_tensor  * weights,
        काष्ठा ggml_tensor  * mask) {

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
    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    परिणाम->op   = GGML_OP_LIGHTNING_INDEXER;
    परिणाम->src[0] = q;
    परिणाम->src[1] = k;
    परिणाम->src[2] = weights;
    परिणाम->src[3] = mask;

    वापस परिणाम;
}

// ggml_dsv4_hc_comb

काष्ठा ggml_tensor * ggml_dsv4_hc_comb(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * mixes,
        काष्ठा ggml_tensor  * scale,
        काष्ठा ggml_tensor  * base,
        भग्न                 eps,
        int32_t               n_iter) {
    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(n_iter > 0);

    स्थिर int64_t hc_mix_dim = mixes->ne[0];
    स्थिर int64_t n_tokens   = mixes->ne[1];

    int64_t hc = 0;
    क्रम (int64_t i = 1; i*i + 2*i <= hc_mix_dim; ++i) {
        अगर ((2 + i)*i == hc_mix_dim) {
            hc = i;
            अवरोध;
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

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);

    ggml_set_op_params_f32(परिणाम, 0, eps);
    ggml_set_op_params_i32(परिणाम, 1, n_iter);

    परिणाम->op     = GGML_OP_DSV4_HC_COMB;
    परिणाम->src[0] = mixes;
    परिणाम->src[1] = scale;
    परिणाम->src[2] = base;

    वापस परिणाम;
}

// ggml_dsv4_hc_pre

अटल काष्ठा ggml_tensor * ggml_dsv4_hc_pre_impl(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * x,
        काष्ठा ggml_tensor  * weights,
        भग्न                 scale,
        bool                  gated) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t hc       = x->ne[1];
    स्थिर int64_t n_tokens = x->ne[2];

    GGML_ASSERT(hc > 0);
    GGML_ASSERT(x->ne[3] == 1);
    अगर (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        GGML_ASSERT(weights->ne[2] == 1);
    }
    GGML_ASSERT(weights->ne[3] == 1);

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

    ggml_set_op_params_f32(परिणाम, 0, scale);
    ggml_set_op_params_i32(परिणाम, 1, gated ? 1 : 0);

    परिणाम->op     = GGML_OP_DSV4_HC_PRE;
    परिणाम->src[0] = x;
    परिणाम->src[1] = weights;

    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_dsv4_hc_pre(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * x,
        काष्ठा ggml_tensor  * weights) {
    वापस ggml_dsv4_hc_pre_impl(ctx, x, weights, 1.0f, false);
}

काष्ठा ggml_tensor * ggml_dsv4_hc_pre_gated(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * x,
        काष्ठा ggml_tensor  * gate,
        भग्न                 scale) {
    वापस ggml_dsv4_hc_pre_impl(ctx, x, gate, scale, true);
}

// ggml_dsv4_hc_post

काष्ठा ggml_tensor * ggml_dsv4_hc_post(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * x,
        काष्ठा ggml_tensor  * residual,
        काष्ठा ggml_tensor  * post,
        काष्ठा ggml_tensor  * comb) {
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t n_tokens = x->ne[1];
    स्थिर int64_t hc       = residual->ne[1];

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

    अगर (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        GGML_ASSERT(comb->ne[3] == 1);
    }

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);

    परिणाम->op     = GGML_OP_DSV4_HC_POST;
    परिणाम->src[0] = x;
    परिणाम->src[1] = residual;
    परिणाम->src[2] = post;
    परिणाम->src[3] = comb;

    वापस परिणाम;
}

// ggml_top_k_mask

काष्ठा ggml_tensor * ggml_शीर्ष_k_mask(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_tensor  * mask,
        काष्ठा ggml_tensor  * शीर्ष_k,
        काष्ठा ggml_tensor  * score,
        काष्ठा ggml_tensor  * blocks,
        काष्ठा ggml_tensor  * extra) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->nb[0] == ggml_type_आकार(mask->type));
    GGML_ASSERT(शीर्ष_k->type == GGML_TYPE_I32);
    GGML_ASSERT(शीर्ष_k->ne[1] % mask->ne[1] == 0 && शीर्ष_k->ne[2] % mask->ne[2] == 0 && शीर्ष_k->ne[3] % mask->ne[3] == 0);

    अगर (score) {
        GGML_ASSERT(score->type == GGML_TYPE_F32);
        GGML_ASSERT(score->ne[1] == शीर्ष_k->ne[1] && score->ne[2] == शीर्ष_k->ne[2] && score->ne[3] == शीर्ष_k->ne[3]);
    }

    अगर (blocks) {
        GGML_ASSERT(blocks->type == GGML_TYPE_I32);
        GGML_ASSERT(blocks->ne[2] == 1 && blocks->ne[3] == 1);
        GGML_ASSERT(!score || blocks->ne[1] == score->ne[0]);
    }

    अगर (extra) {
        GGML_ASSERT(extra->type == GGML_TYPE_I32);
        GGML_ASSERT(extra->ne[1] == शीर्ष_k->ne[1] && extra->ne[2] == शीर्ष_k->ne[2] && extra->ne[3] == शीर्ष_k->ne[3]);
    }

    काष्ठा ggml_tensor * परिणाम = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], शीर्ष_k->ne[1], शीर्ष_k->ne[2], शीर्ष_k->ne[3]);

    परिणाम->op     = GGML_OP_शीर्ष_K_MASK;
    परिणाम->src[0] = mask;
    परिणाम->src[1] = शीर्ष_k;
    परिणाम->src[2] = score;
    परिणाम->src[3] = blocks;
    परिणाम->src[4] = extra;

    वापस परिणाम;
}

////////////////////////////////////////////////////////////////////////////////

काष्ठा ggml_hash_set ggml_hash_set_new(माप_प्रकार आकार) {
    आकार = ggml_hash_आकार(आकार);
    काष्ठा ggml_hash_set परिणाम;
    परिणाम.आकार = आकार;
    परिणाम.कुंजियाँ = GGML_MALLOC(माप(काष्ठा ggml_tensor *) * आकार);
    परिणाम.used = GGML_CALLOC(ggml_bitset_आकार(आकार), माप(ggml_bitset_t));
    वापस परिणाम;
}

व्योम ggml_hash_set_reset(काष्ठा ggml_hash_set * hash_set) {
    स_रखो(hash_set->used, 0, माप(ggml_bitset_t) * ggml_bitset_आकार(hash_set->आकार));
}

व्योम ggml_hash_set_free(काष्ठा ggml_hash_set * hash_set) {
    GGML_FREE(hash_set->used);
    GGML_FREE(hash_set->कुंजियाँ);
}

माप_प्रकार ggml_hash_आकार(माप_प्रकार min_sz) {
    // next primes after powers of two
    अटल स्थिर माप_प्रकार primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    अटल स्थिर माप_प्रकार n_primes = माप(primes)/माप(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    माप_प्रकार l = 0;
    माप_प्रकार r = n_primes;
    जबतक (l < r) {
        माप_प्रकार m = (l + r)/2;
        अगर (primes[m] < min_sz) {
            l = m + 1;
        } अन्यथा {
            r = m;
        }
    }
    माप_प्रकार sz = l < n_primes ? primes[l] : min_sz | 1;
    वापस sz;
}

काष्ठा hash_map {
    काष्ठा ggml_hash_set set;
    काष्ठा ggml_tensor ** vals;
};

अटल काष्ठा hash_map * ggml_new_hash_map(माप_प्रकार आकार) {
    काष्ठा hash_map * परिणाम = GGML_MALLOC(माप(काष्ठा hash_map));
    परिणाम->set = ggml_hash_set_new(आकार);
    परिणाम->vals = GGML_CALLOC(परिणाम->set.आकार, माप(काष्ठा ggml_tensor *));
    वापस परिणाम;
}

अटल व्योम ggml_hash_map_free(काष्ठा hash_map * map) {
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

अटल व्योम ggml_add_or_set(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_cgraph  * cgraph,
        माप_प्रकार                isrc,
        काष्ठा ggml_tensor  * tensor) {
    काष्ठा ggml_tensor * src = cgraph->visited_hash_set.कुंजियाँ[isrc];
    GGML_ASSERT(src);
    अगर (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add_impl(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = tensor;
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

अटल व्योम ggml_acc_or_set(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_cgraph  * cgraph,
        माप_प्रकार                isrc,
        काष्ठा ggml_tensor  * tensor,
        स्थिर  माप_प्रकार         nb1,
        स्थिर  माप_प्रकार         nb2,
        स्थिर  माप_प्रकार         nb3,
        स्थिर  माप_प्रकार         offset) {
    काष्ठा ggml_tensor * src = cgraph->visited_hash_set.कुंजियाँ[isrc];
    GGML_ASSERT(src);
    अगर (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_acc_impl(ctx, cgraph->grads[isrc], tensor, nb1, nb2, nb3, offset, cgraph->grad_accs[isrc]);
    } अन्यथा {
        काष्ठा ggml_tensor * a_zero = ggml_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        cgraph->grads[isrc] = ggml_acc_impl(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.कुंजियाँ[isrc]->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

अटल व्योम ggml_add1_or_set(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_cgraph  * cgraph,
        माप_प्रकार                isrc,
        काष्ठा ggml_tensor  * tensor) {
    काष्ठा ggml_tensor * src = cgraph->visited_hash_set.कुंजियाँ[isrc];
    GGML_ASSERT(src);
    अगर (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = ggml_repeat(ctx, tensor, src);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

अटल व्योम ggml_sub_or_set(
        काष्ठा ggml_context * ctx,
        काष्ठा ggml_cgraph  * cgraph,
        माप_प्रकार                isrc,
        काष्ठा ggml_tensor  * tensor) {
    काष्ठा ggml_tensor * src = cgraph->visited_hash_set.कुंजियाँ[isrc];
    GGML_ASSERT(src);
    अगर (cgraph->grads[isrc]) {
        cgraph->grads[isrc] = ggml_sub_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
    } अन्यथा {
        cgraph->grads[isrc] = ggml_neg(ctx, tensor);
    }
    ggml_format_name(cgraph->grads[isrc], "grad for %s", src->name);
    ggml_build_forward_expand(cgraph, cgraph->grads[isrc]);
}

अटल व्योम ggml_compute_backward(
        काष्ठा ggml_context * ctx, काष्ठा ggml_cgraph * cgraph, पूर्णांक i, स्थिर bool * grads_needed) {
    काष्ठा ggml_tensor * tensor = cgraph->nodes[i];
    काष्ठा ggml_tensor * grad   = ggml_graph_get_grad(cgraph, tensor);

    अगर (!grad) {
        वापस;
    }

    काष्ठा ggml_tensor * src0 = tensor->src[0];
    काष्ठा ggml_tensor * src1 = tensor->src[1];
    काष्ठा ggml_tensor * src2 = tensor->src[2];
    काष्ठा ggml_hash_set * hash_set = &cgraph->visited_hash_set;
    स्थिर माप_प्रकार isrc0 = src0 ? ggml_hash_find(hash_set, src0) : (माप_प्रकार) -1;
    स्थिर माप_प्रकार isrc1 = src1 ? ggml_hash_find(hash_set, src1) : (माप_प्रकार) -1;
    स्थिर माप_प्रकार isrc2 = src2 ? ggml_hash_find(hash_set, src2) : (माप_प्रकार) -1;
    स्थिर bool src0_needs_grads = src0 && isrc0 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc0) && grads_needed[isrc0];
    स्थिर bool src1_needs_grads = src1 && isrc1 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc1) && grads_needed[isrc1];
    स्थिर bool src2_needs_grads = src2 && isrc2 != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, isrc2) && grads_needed[isrc2];

    चयन (tensor->op) {
        हाल GGML_OP_DUP: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
        } अवरोध;
        हाल GGML_OP_ADD: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            अगर (src1_needs_grads) {
                काष्ठा ggml_tensor * tmp = grad;
                अगर (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } अवरोध;
        हाल GGML_OP_ADD1: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            अगर (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_mean(ctx, grad)); // TODO: should probably be sum instead of mean
            }
        } अवरोध;
        हाल GGML_OP_ACC: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            अगर (src1_needs_grads) {
                स्थिर माप_प्रकार nb1    = ((int32_t *) tensor->op_params)[0];
                स्थिर माप_प्रकार nb2    = ((int32_t *) tensor->op_params)[1];
                स्थिर माप_प्रकार nb3    = ((int32_t *) tensor->op_params)[2];
                स्थिर माप_प्रकार offset = ((int32_t *) tensor->op_params)[3];

                काष्ठा ggml_tensor * tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } अवरोध;
        हाल GGML_OP_SUB: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, grad);
            }
            अगर (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, grad);
            }
        } अवरोध;
        हाल GGML_OP_MUL: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, src1));
            }
            अगर (src1_needs_grads) {
                काष्ठा ggml_tensor * tmp = ggml_mul(ctx, src0, grad);
                अगर (!ggml_are_same_shape(src0, src1)) {
                    tmp = ggml_repeat_back(ctx, tmp, src1);
                }
                ggml_add_or_set(ctx, cgraph, isrc1, tmp);
            }
        } अवरोध;
        हाल GGML_OP_DIV: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src1));
            }
            अगर (src1_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, grad, ggml_div(ctx, tensor, src1)));
            }
        } अवरोध;
        हाल GGML_OP_SQR: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_mul(ctx, src0, grad), 2.0f));
            }
        } अवरोध;
        हाल GGML_OP_SQRT: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale(ctx, ggml_div(ctx, grad, tensor), 0.5f));
            }
        } अवरोध;
        हाल GGML_OP_LOG: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_div(ctx, grad, src0));
            }
        } अवरोध;
        हाल GGML_OP_SIN: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_cos(ctx, src0)));
            }
        } अवरोध;
        हाल GGML_OP_COS: {
            अगर (src0_needs_grads) {
                ggml_sub_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sin(ctx, src0)));
            }
        } अवरोध;
        हाल GGML_OP_योग: {
            अगर (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, grad);
            }
        } अवरोध;
        हाल GGML_OP_योग_ROWS: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } अवरोध;
        हाल GGML_OP_MEAN: {
            अगर (src0_needs_grads) {
                ggml_add1_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, 1.0f/src0->ne[0], 0.0, false));
            }
        } अवरोध;
        हाल GGML_OP_REPEAT: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat_back(ctx, grad, src0));
            }
        } अवरोध;
        हाल GGML_OP_REPEAT_BACK: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_repeat(ctx, grad, src0));
            }
        } अवरोध;
        हाल GGML_OP_RMS_NORM: {
            अगर (src0_needs_grads) {
                भग्न eps;
                स_नकल(&eps, tensor->op_params, माप(भग्न));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_rms_norm_back(ctx, grad, src0, eps));
            }
        } अवरोध;
        हाल GGML_OP_MUL_MAT: {
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

            अगर (src0_needs_grads) {
                GGML_ASSERT(grad->ne[2] == src1->ne[2]);
                GGML_ASSERT(grad->ne[3] == src1->ne[3]);
                काष्ठा ggml_tensor * tmp =
                    ggml_out_prod(ctx, // [n,m,qq,rr]
                        src1,          // [n,p,qq,rr]
                        grad);         // [m,p,qq,rr]
                अगर (!ggml_are_same_shape(tmp, src0)) {
                    GGML_ASSERT(tmp->ne[0] == src0->ne[0]);
                    GGML_ASSERT(tmp->ne[1] == src0->ne[1]);
                    GGML_ASSERT(tmp->ne[3] == 1);

                    स्थिर int64_t nr2 = tmp->ne[2] / src0->ne[2];
                    स्थिर माप_प्रकार nb2 = tmp->nb[2] * nr2;
                    स्थिर माप_प्रकार nb3 = tmp->nb[2];

                    tmp = ggml_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
                    tmp = ggml_repeat_back(ctx, tmp, src0);
                }
                ggml_add_or_set(ctx, cgraph, isrc0, tmp);
            }
            अगर (src1_needs_grads) {
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
        } अवरोध;
        हाल GGML_OP_SCALE: {
            अगर (src0_needs_grads) {
                भग्न s;
                स_नकल(&s, tensor->op_params, माप(भग्न));
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_scale_impl(ctx, grad, s, 0.0, false));
            }
        } अवरोध;
        हाल GGML_OP_SET: {
            स्थिर माप_प्रकार nb1    = ((स्थिर int32_t *) tensor->op_params)[0];
            स्थिर माप_प्रकार nb2    = ((स्थिर int32_t *) tensor->op_params)[1];
            स्थिर माप_प्रकार nb3    = ((स्थिर int32_t *) tensor->op_params)[2];
            स्थिर माप_प्रकार offset = ((स्थिर int32_t *) tensor->op_params)[3];

            काष्ठा ggml_tensor * tensor_grad_view = शून्य;

            अगर (src0_needs_grads || src1_needs_grads) {
                GGML_ASSERT(src0->type == tensor->type);
                GGML_ASSERT(!cgraph->grads[isrc0] ||                      cgraph->grads[isrc0]->type == grad->type);
                GGML_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

                tensor_grad_view = ggml_view_4d(ctx,
                    grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                    nb1, nb2, nb3, offset);
            }

            अगर (src0_needs_grads) {
                काष्ठा ggml_tensor * tmp = ggml_neg(ctx, tensor_grad_view);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_acc_impl(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
            }

            अगर (src1_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc1, ggml_reshape(ctx, ggml_cont(ctx, tensor_grad_view), src1));
            }
        } अवरोध;
        हाल GGML_OP_CPY: {
            // cpy overwrites value of src1 by src0 and returns view(src1)
            // the overwriting is mathematically equivalent to:
            // tensor = src0 * 1 + src1 * 0
            अगर (src0_needs_grads) {
                // dsrc0 = dtensor * 1
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad, src0));
            }
            अगर (src1_needs_grads) {
                // dsrc1 = dtensor * 0 -> noop
            }
        } अवरोध;
        हाल GGML_OP_CONT: {
            // same as cpy
            अगर (src0_needs_grads) {
                GGML_ASSERT(!cgraph->grads[isrc0] || ggml_is_contiguous(cgraph->grads[isrc0]));
                GGML_ASSERT(ggml_is_contiguous(grad));
                GGML_ASSERT(ggml_nelements(tensor) == ggml_nelements(src0));
                ggml_add_or_set(ctx, cgraph, isrc0,
                    ggml_are_same_shape(tensor, src0) ? grad : ggml_reshape(ctx, grad, src0));
            }
        } अवरोध;
        हाल GGML_OP_RESHAPE: {
            अगर (src0_needs_grads) {
                काष्ठा ggml_tensor * grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad);
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_reshape(ctx, grad_cont, src0));
            }
        } अवरोध;
        हाल GGML_OP_VIEW: {
            अगर (src0_needs_grads) {
                माप_प्रकार offset;

                स_नकल(&offset, tensor->op_params, माप(offset));

                माप_प्रकार nb1 = tensor->nb[1];
                माप_प्रकार nb2 = tensor->nb[2];
                माप_प्रकार nb3 = tensor->nb[3];

                अगर (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
                    // gradient is typically F32, but src0 could be other type
                    माप_प्रकार ng = ggml_element_आकार(cgraph->grads[isrc0]);
                    माप_प्रकार n0 = ggml_element_आकार(src0);
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
        } अवरोध;
        हाल GGML_OP_PERMUTE: {
            अगर (src0_needs_grads) {
                स्थिर int32_t * axes = (स्थिर int32_t *) tensor->op_params;
                स्थिर पूर्णांक axis0 = axes[0] & 0x3;
                स्थिर पूर्णांक axis1 = axes[1] & 0x3;
                स्थिर पूर्णांक axis2 = axes[2] & 0x3;
                स्थिर पूर्णांक axis3 = axes[3] & 0x3;
                पूर्णांक axb[4] = {0,0,0,0}; // axes backward
                axb[axis0] = 0;
                axb[axis1] = 1;
                axb[axis2] = 2;
                axb[axis3] = 3;
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
            }
        } अवरोध;
        हाल GGML_OP_TRANSPOSE: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_transpose(ctx, grad));
            }
        } अवरोध;
        हाल GGML_OP_GET_ROWS: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_get_rows_back(ctx, grad, src1, src0));
            }
            अगर (src1_needs_grads) {
                // noop
            }
        } अवरोध;
        हाल GGML_OP_DIAG_MASK_INF: {
            अगर (src0_needs_grads) {
                /* ggml_diag_mask_inf_impl() shouldn't be here */
                /* ref:  https://github.com/ggml-org/llama.cpp/pull/4203#discussion_r1412377992 */
                स्थिर पूर्णांक n_past = ((स्थिर int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } अवरोध;
        हाल GGML_OP_DIAG_MASK_ZERO: {
            अगर (src0_needs_grads) {
                स्थिर पूर्णांक n_past = ((स्थिर int32_t *) tensor->op_params)[0];
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_diag_mask_zero_impl(ctx, grad, n_past, false));
            }
        } अवरोध;
        हाल GGML_OP_SOFT_MAX: {
            अगर (src0_needs_grads) {
                भग्न scale    = 1.0f;
                भग्न max_bias = 0.0f;

                स_नकल(&scale,    (स्थिर भग्न *) tensor->op_params + 0, माप(भग्न));
                स_नकल(&max_bias, (स्थिर भग्न *) tensor->op_params + 1, माप(भग्न));

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
            }
            GGML_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
        } अवरोध;
        हाल GGML_OP_ROPE: {
            अगर (src0_needs_grads) {
                //const int n_past = ((int32_t *) tensor->op_params)[0];
                स्थिर पूर्णांक n_dims     = ((स्थिर int32_t *) tensor->op_params)[1];
                स्थिर पूर्णांक mode       = ((स्थिर int32_t *) tensor->op_params)[2];
                //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                स्थिर पूर्णांक n_ctx_orig = ((स्थिर int32_t *) tensor->op_params)[4];
                भग्न freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                पूर्णांक sections[4] = {0, 0, 0, 0};

                स_नकल(&freq_base,   (स्थिर भग्न *) tensor->op_params +  5, माप(भग्न));
                स_नकल(&freq_scale,  (स्थिर भग्न *) tensor->op_params +  6, माप(भग्न));
                स_नकल(&ext_factor,  (स्थिर भग्न *) tensor->op_params +  7, माप(भग्न));
                स_नकल(&attn_factor, (स्थिर भग्न *) tensor->op_params +  8, माप(भग्न));
                स_नकल(&beta_fast,   (स्थिर भग्न *) tensor->op_params +  9, माप(भग्न));
                स_नकल(&beta_slow,   (स्थिर भग्न *) tensor->op_params + 10, माप(भग्न));
                स_नकल(&sections,                    tensor->op_params + 11, माप(sections));

                काष्ठा ggml_tensor * rope_back = grad->ne[2] == src1->ne[0] ?
                    ggml_rope_ext_back(ctx, grad, src1, src2, n_dims,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow) :
                    ggml_rope_multi_back(ctx, grad, src1, src2, n_dims, sections,
                        mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ggml_add_or_set(ctx, cgraph, isrc0, rope_back);
            }
            GGML_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
        } अवरोध;
        हाल GGML_OP_IM2COL: {
            अगर (src1_needs_grads) {
                स्थिर int32_t s0    = ggml_get_op_params_i32(tensor, 0);
                स्थिर int32_t s1    = ggml_get_op_params_i32(tensor, 1);
                स्थिर int32_t p0    = ggml_get_op_params_i32(tensor, 2);
                स्थिर int32_t p1    = ggml_get_op_params_i32(tensor, 3);
                स्थिर int32_t d0    = ggml_get_op_params_i32(tensor, 4);
                स्थिर int32_t d1    = ggml_get_op_params_i32(tensor, 5);
                स्थिर bool    is_2D = ggml_get_op_params_i32(tensor, 6) == 1;

                ggml_add_or_set(ctx, cgraph, isrc1, ggml_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
            }
        } अवरोध;
        हाल GGML_OP_POOL_2D: {
            अगर (src0_needs_grads) {
                स्थिर क्रमागत ggml_op_pool op = ggml_get_op_params_i32(tensor, 0);
                स्थिर      int32_t      k0 = ggml_get_op_params_i32(tensor, 1);
                स्थिर      int32_t      k1 = ggml_get_op_params_i32(tensor, 2);
                स्थिर      int32_t      s0 = ggml_get_op_params_i32(tensor, 3);
                स्थिर      int32_t      s1 = ggml_get_op_params_i32(tensor, 4);
                स्थिर      int32_t      p0 = ggml_get_op_params_i32(tensor, 5);
                स्थिर      int32_t      p1 = ggml_get_op_params_i32(tensor, 6);

                ggml_add_or_set(ctx, cgraph, isrc0, ggml_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
            }
        } अवरोध;
        हाल GGML_OP_WIN_PART:
        हाल GGML_OP_WIN_UNPART:
        हाल GGML_OP_UNARY: {
            चयन (ggml_get_unary_op(tensor)) {
                हाल GGML_UNARY_OP_ABS: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_sgn(ctx, src0), grad));
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_SGN: {
                    // noop
                } अवरोध;
                हाल GGML_UNARY_OP_NEG: {
                    अगर (src0_needs_grads) {
                        ggml_sub_or_set(ctx, cgraph, isrc0, grad);
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_STEP: {
                    // noop
                } अवरोध;
                हाल GGML_UNARY_OP_RELU: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, ggml_step(ctx, src0), grad));
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_SILU: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, grad, src0));
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_EXP: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, tensor, grad));
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_EXPM1: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_exp(ctx, src0)));
                    }
                } अवरोध;
                हाल GGML_UNARY_OP_SOFTPLUS: {
                    अगर (src0_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_mul(ctx, grad, ggml_sigmoid(ctx, src0)));
                    }
                } अवरोध;
                शेष: {
                    ख_लिखो(मानक_त्रुटि, "%s: unsupported unary op for backward pass: %s\n",
                        __func__, ggml_unary_op_name(ggml_get_unary_op(tensor)));
                    GGML_ABORT("fatal error");
                } //break;
            }
        } अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS: {
            अगर (src0_needs_grads) {
                ggml_add_or_set(ctx, cgraph, isrc0, ggml_cross_entropy_loss_back(ctx, grad, src0, src1));
            }
            GGML_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
        } अवरोध;
        हाल GGML_OP_GLU: {
            चयन (ggml_get_glu_op(tensor)) {
                हाल GGML_GLU_OP_SWIGLU: {
                    अगर (src0_needs_grads) {
                        GGML_ASSERT(src1 && "backward pass only implemented for split swiglu");
                        ggml_add_or_set(ctx, cgraph, isrc0, ggml_silu_back(ctx, ggml_mul(ctx, grad, src1), src0));
                    }
                    अगर (src1_needs_grads) {
                        ggml_add_or_set(ctx, cgraph, isrc1, ggml_mul(ctx, ggml_silu(ctx, src0), grad));
                    }
                } अवरोध;
                शेष: {
                    GGML_ABORT("unsupported glu op for backward pass: %s", ggml_glu_op_name(ggml_get_glu_op(tensor)));
                } //break;
            }
        } अवरोध;
        हाल GGML_OP_NONE: {
            // noop
        } अवरोध;
        हाल GGML_OP_गिनती:
        शेष: {
            GGML_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, ggml_op_name(tensor->op));
        } //break;
    }

    GGML_ASSERT(!src0_needs_grads || ggml_are_same_shape(src0, cgraph->grads[isrc0]));
    GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]));
    GGML_ASSERT(!src2_needs_grads || ggml_are_same_shape(src2, cgraph->grads[isrc2]));
}

अटल माप_प्रकार ggml_visit_parents_graph(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_tensor * गाँठ, bool compute) {
    अगर (गाँठ->op != GGML_OP_NONE && compute) {
        गाँठ->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    स्थिर माप_प्रकार गाँठ_hash_pos = ggml_hash_find(&cgraph->visited_hash_set, गाँठ);
    GGML_ASSERT(गाँठ_hash_pos != GGML_HASHSET_FULL);

    अगर (ggml_bitset_get(cgraph->visited_hash_set.used, गाँठ_hash_pos)) {
        // already visited

        अगर (compute) {
            // update the compute flag regardless
            क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; ++i) {
                काष्ठा ggml_tensor * src = गाँठ->src[i];
                अगर (src && ((src->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)) {
                    ggml_visit_parents_graph(cgraph, src, true);
                }
            }
        }

        वापस गाँठ_hash_pos;
    }

    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.कुंजियाँ[गाँठ_hash_pos] = गाँठ;
    ggml_bitset_set(cgraph->visited_hash_set.used, गाँठ_hash_pos);
    cgraph->use_counts[गाँठ_hash_pos] = 0;

    क्रम (पूर्णांक i = 0; i < GGML_MAX_SRC; ++i) {
        स्थिर पूर्णांक k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_बायाँ_TO_दायाँ) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_दायाँ_TO_बायाँ) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i */ i;

        काष्ठा ggml_tensor * src = गाँठ->src[k];
        अगर (src) {
            स्थिर माप_प्रकार src_hash_pos = ggml_visit_parents_graph(cgraph, src, compute);

            // Update the use count for this operand.
            cgraph->use_counts[src_hash_pos]++;
        }
    }

    अगर (गाँठ->op == GGML_OP_NONE && !(गाँठ->flags & GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->आकार);

        अगर (म_माप(गाँठ->name) == 0) {
            ggml_format_name(गाँठ, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = गाँठ;
        cgraph->n_leafs++;
    } अन्यथा {
        GGML_ASSERT(cgraph->n_nodes < cgraph->आकार);

        अगर (म_माप(गाँठ->name) == 0) {
            ggml_format_name(गाँठ, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = गाँठ;
        cgraph->n_nodes++;
    }

    वापस गाँठ_hash_pos;
}

अटल व्योम ggml_build_forward_impl(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_tensor * tensor, bool expand, bool compute) {
    अगर (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    स्थिर पूर्णांक n_old = cgraph->n_nodes;

    ggml_visit_parents_graph(cgraph, tensor, compute);

    स्थिर पूर्णांक n_new = cgraph->n_nodes - n_old;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    अगर (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

काष्ठा ggml_tensor * ggml_build_forward_select(
        काष्ठा ggml_cgraph  * cgraph,
        काष्ठा ggml_tensor ** tensors,
        पूर्णांक                   n_tensors,
        पूर्णांक                   idx) {
    GGML_ASSERT(idx >= 0 && idx < n_tensors);

    क्रम (पूर्णांक i = 0; i < n_tensors; i++) {
        ggml_build_forward_impl(cgraph, tensors[i], true, i == idx ? true : false);
    }

    वापस tensors[idx];
}

व्योम ggml_build_forward_expand(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, true);
}

व्योम ggml_build_forward_order(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true, false);
}

व्योम ggml_build_backward_expand(
        काष्ठा ggml_context *  ctx,
        काष्ठा ggml_cgraph  *  cgraph,
        काष्ठा ggml_tensor  ** grad_accs) {
    GGML_ASSERT(cgraph->n_nodes > 0);
    GGML_ASSERT(cgraph->grads);
    GGML_ASSERT(cgraph->grad_accs);

    स्थिर पूर्णांक n_nodes_f = cgraph->n_nodes;

    स_रखो(cgraph->grads,     0, cgraph->visited_hash_set.आकार*माप(काष्ठा ggml_tensor *));
    स_रखो(cgraph->grad_accs, 0, cgraph->visited_hash_set.आकार*माप(काष्ठा ggml_tensor *));
    bool * grads_needed = सुस्मृति(cgraph->visited_hash_set.आकार, माप(bool));

    {
        bool any_params = false;
        bool any_loss   = false;
        क्रम (पूर्णांक i = 0; i < n_nodes_f; ++i) {
            काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];
            any_params = any_params || (गाँठ->flags & GGML_TENSOR_FLAG_PARAM);
            any_loss   = any_loss   || (गाँठ->flags & GGML_TENSOR_FLAG_LOSS);
        }
        GGML_ASSERT(any_params && "no trainable parameters found, did you forget to call ggml_set_param?");
        GGML_ASSERT(any_loss && "no training loss found, did you forget to call ggml_set_loss?");
    }

    क्रम (पूर्णांक i = 0; i < n_nodes_f; ++i) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        अगर (गाँठ->type == GGML_TYPE_I32) {
            जारी;
        }

        bool गाँठ_needs_grad = (गाँठ->flags & GGML_TENSOR_FLAG_PARAM) || (गाँठ->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        चयन (गाँठ->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            हाल GGML_OP_IM2COL:      // only used for its shape
            हाल GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                अवरोध;
            हाल GGML_OP_UNARY: {
                स्थिर क्रमागत ggml_unary_op uop = ggml_get_unary_op(गाँठ);
                // SGN and STEP unary ops are piecewise constant
                अगर (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } अवरोध;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            हाल GGML_OP_CPY:           // gradients in CPY target are irrelevant
            हाल GGML_OP_GET_ROWS:      // row indices not differentiable
            हाल GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            हाल GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                अवरोध;

            शेष:
                अवरोध;
        }
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; ++j) {
            अगर (!गाँठ->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, गाँठ->src[j])]) {
                जारी;
            }
            GGML_ASSERT(गाँठ->src[j]->type == GGML_TYPE_F32 || गाँठ->src[j]->type == GGML_TYPE_F16);
            गाँठ_needs_grad = true;
            अवरोध;
        }
        अगर (!गाँठ_needs_grad) {
            जारी;
        }

        // inplace operations are currently not supported
        GGML_ASSERT(!गाँठ->view_src || गाँठ->op == GGML_OP_CPY || गाँठ->op == GGML_OP_SET_ROWS || गाँठ->op == GGML_OP_VIEW ||
            गाँठ->op == GGML_OP_RESHAPE || गाँठ->op == GGML_OP_PERMUTE || गाँठ->op == GGML_OP_TRANSPOSE);

        स्थिर माप_प्रकार ihash = ggml_hash_find(&cgraph->visited_hash_set, गाँठ);
        GGML_ASSERT(ihash != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(cgraph->visited_hash_set.used, ihash));
        अगर (grad_accs && grad_accs[i]) {
            cgraph->grad_accs[ihash] = grad_accs[i];
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        } अन्यथा अगर (गाँठ->flags & GGML_TENSOR_FLAG_LOSS) {
            // loss tensors always need a gradient accumulator
            cgraph->grad_accs[ihash] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, गाँठ->ne);
            cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
        }
        grads_needed[ihash] = true;
    }

    क्रम (पूर्णांक i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        ggml_compute_backward(ctx, cgraph, i, grads_needed);
    }

    मुक्त(grads_needed);
}

अटल व्योम * incr_ptr_aligned(व्योम ** p, माप_प्रकार आकार, माप_प्रकार align) {
    व्योम * ptr = *p;
    ptr = (व्योम *) GGML_PAD((uintptr_t) ptr, align);
    *p = (व्योम *) ((अक्षर *) ptr + आकार);
    वापस ptr;
}

अटल माप_प्रकार ggml_graph_nbytes(माप_प्रकार आकार, bool grads) {
    माप_प्रकार hash_आकार = ggml_hash_आकार(आकार * 2);
    व्योम * p = (अक्षर *) 1024; // workaround for ubsan error "applying non-zero offset X to null pointer"
    incr_ptr_aligned(&p, माप(काष्ठा ggml_cgraph), 1);
    incr_ptr_aligned(&p, आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_आकार * माप(int32_t), माप(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)); // hash keys
    अगर (grads) {
        incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)); // grad_accs
    }
    incr_ptr_aligned(&p, ggml_bitset_आकार(hash_आकार) * माप(ggml_bitset_t), माप(ggml_bitset_t));

    माप_प्रकार nbytes = (माप_प्रकार) p;
    वापस nbytes - 1024;
}

माप_प्रकार ggml_graph_overhead_custom(माप_प्रकार आकार, bool grads) {
    वापस GGML_OBJECT_आकार + GGML_PAD(ggml_graph_nbytes(आकार, grads), GGML_MEM_ALIGN);
}

माप_प्रकार ggml_graph_overhead(व्योम) {
    वापस ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_आकार, false);
}

काष्ठा ggml_cgraph * ggml_new_graph_custom(काष्ठा ggml_context * ctx, माप_प्रकार आकार, bool grads) {
    स्थिर माप_प्रकार obj_आकार = ggml_graph_nbytes(आकार, grads);
    काष्ठा ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_आकार);
    काष्ठा ggml_cgraph * cgraph = (काष्ठा ggml_cgraph *) ((अक्षर *) ctx->mem_बफर + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    माप_प्रकार hash_आकार = ggml_hash_आकार(आकार * 2);

    व्योम * p = cgraph + 1;

    काष्ठा ggml_tensor ** nodes_ptr      =         incr_ptr_aligned(&p, आकार      * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *));
    काष्ठा ggml_tensor ** leafs_ptr      =         incr_ptr_aligned(&p, आकार      * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *));
    int32_t             * use_counts_ptr =         incr_ptr_aligned(&p, hash_आकार * माप(int32_t), माप(int32_t));
    काष्ठा ggml_tensor ** hash_कुंजियाँ_ptr  =         incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *));
    काष्ठा ggml_tensor ** grads_ptr      = grads ? incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)) : शून्य;
    काष्ठा ggml_tensor ** grad_accs_ptr  = grads ? incr_ptr_aligned(&p, hash_आकार * माप(काष्ठा ggml_tensor *), माप(काष्ठा ggml_tensor *)) : शून्य;

    ggml_bitset_t * hash_used = incr_ptr_aligned(&p, ggml_bitset_आकार(hash_आकार) * माप(ggml_bitset_t), माप(ggml_bitset_t));

    // check that we allocated the correct amount of memory
    निश्चित(obj_आकार == (माप_प्रकार)((अक्षर *)p - (अक्षर *)cgraph));

    *cgraph = (काष्ठा ggml_cgraph) {
        /*.size         =*/ आकार,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.grad_accs    =*/ grad_accs_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.use_counts   =*/ use_counts_ptr,
        /*.hash_table   =*/ { hash_आकार, hash_used, hash_कुंजियाँ_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_बायाँ_TO_दायाँ,
        /*.uid          =*/ 0,
    };

    ggml_hash_set_reset(&cgraph->visited_hash_set);
    अगर (grads) {
        स_रखो(cgraph->grads,     0, hash_आकार*माप(काष्ठा ggml_tensor *));
        स_रखो(cgraph->grad_accs, 0, hash_आकार*माप(काष्ठा ggml_tensor *));
    }

    वापस cgraph;
}

काष्ठा ggml_cgraph * ggml_new_graph(काष्ठा ggml_context * ctx) {
    वापस ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_आकार, false);
}

काष्ठा ggml_cgraph ggml_graph_view(काष्ठा ggml_cgraph * cgraph0, पूर्णांक i0, पूर्णांक i1) {
    काष्ठा ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ शून्य, // gradients would need visited_hash_set
        /*.grad_accs        =*/ शून्य,
        /*.leafs            =*/ शून्य,
        /*.use_counts       =*/ cgraph0->use_counts,
        /*.visited_hash_set =*/ cgraph0->visited_hash_set,
        /*.order            =*/ cgraph0->order,
        /*.uid              =*/ 0
    };

    वापस cgraph;
}

व्योम ggml_graph_cpy(काष्ठा ggml_cgraph * src, काष्ठा ggml_cgraph * dst) {
    GGML_ASSERT(dst->आकार >= src->n_leafs);
    GGML_ASSERT(dst->आकार >= src->n_nodes);
    GGML_ASSERT(dst->visited_hash_set.आकार >= src->visited_hash_set.आकार);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    क्रम (पूर्णांक i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    क्रम (पूर्णांक i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    क्रम (माप_प्रकार i = 0; i < src->visited_hash_set.आकार; ++i) {
        // copy all hashset keys (tensors) that are in use
        अगर (ggml_bitset_get(src->visited_hash_set.used, i)) {
            माप_प्रकार new_hash_pos = ggml_hash_जोड़ो(&dst->visited_hash_set, src->visited_hash_set.कुंजियाँ[i]);
            dst->use_counts[new_hash_pos] = src->use_counts[i];
        }
    }

    अगर (dst->grads) {
        स_रखो(dst->grads,     0, dst->visited_hash_set.आकार*माप(काष्ठा ggml_tensor *));
        स_रखो(dst->grad_accs, 0, dst->visited_hash_set.आकार*माप(काष्ठा ggml_tensor *));
    }
    अगर (src->grads) {
        GGML_ASSERT(dst->grads     != शून्य);
        GGML_ASSERT(dst->grad_accs != शून्य);
        क्रम (पूर्णांक i = 0; i < src->n_nodes; ++i) {
            स्थिर माप_प्रकार igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
            स्थिर माप_प्रकार igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

            GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
            GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
            GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

            dst->grads[igrad_dst]     = src->grads[igrad_src];
            dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
        }
    }
}

काष्ठा ggml_cgraph * ggml_graph_dup(काष्ठा ggml_context * ctx, काष्ठा ggml_cgraph * cgraph, bool force_grads) {
    काष्ठा ggml_cgraph * परिणाम = ggml_new_graph_custom(ctx, cgraph->आकार, cgraph->grads || force_grads);
    ggml_graph_cpy(cgraph, परिणाम);
    वापस परिणाम;
}

काष्ठा ggml_tensor * ggml_set_zero(काष्ठा ggml_tensor * tensor) {
    अगर (ggml_is_empty(tensor)) {
        वापस tensor;
    }
    अगर (tensor->बफर) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    } अन्यथा {
        GGML_ASSERT(tensor->तथ्य);
        स_रखो(tensor->तथ्य, 0, ggml_nbytes(tensor));
    }
    वापस tensor;
}

व्योम ggml_graph_reset(काष्ठा ggml_cgraph * cgraph) {
    अगर (!cgraph) {
        वापस;
    }
    GGML_ASSERT(cgraph->grads != शून्य);

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ     = cgraph->nodes[i];
        काष्ठा ggml_tensor * grad_acc = ggml_graph_get_grad_acc(cgraph, गाँठ);

        अगर (गाँठ->op == GGML_OP_OPT_STEP_ADAMW) {
            // clear momenta
            ggml_set_zero(गाँठ->src[2]);
            ggml_set_zero(गाँठ->src[3]);
        }

        // initial gradients of loss should be 1, 0 otherwise
        अगर (grad_acc) {
            अगर (गाँठ->flags & GGML_TENSOR_FLAG_LOSS) {
                GGML_ASSERT(grad_acc->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_is_scalar(grad_acc));

                स्थिर भग्न onef = 1.0f;
                अगर (grad_acc->बफर) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, माप(भग्न));
                } अन्यथा {
                    GGML_ASSERT(grad_acc->तथ्य);
                    *((भग्न *) grad_acc->तथ्य) = onef;
                }
            } अन्यथा {
                ggml_set_zero(grad_acc);
            }
        }
    }
}

व्योम ggml_graph_clear(काष्ठा ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    ggml_hash_set_reset(&cgraph->visited_hash_set);
}

पूर्णांक ggml_graph_आकार(काष्ठा ggml_cgraph * cgraph) {
    वापस cgraph->आकार;
}

काष्ठा ggml_tensor * ggml_graph_गाँठ(काष्ठा ggml_cgraph * cgraph, पूर्णांक i) {
    अगर (i < 0) {
        GGML_ASSERT(cgraph->n_nodes + i >= 0);
        वापस cgraph->nodes[cgraph->n_nodes + i];
    }

    GGML_ASSERT(i < cgraph->n_nodes);
    वापस cgraph->nodes[i];
}

काष्ठा ggml_tensor ** ggml_graph_nodes(काष्ठा ggml_cgraph * cgraph) {
    वापस cgraph->nodes;
}

पूर्णांक ggml_graph_n_nodes(काष्ठा ggml_cgraph * cgraph) {
    वापस cgraph->n_nodes;
}

व्योम ggml_graph_add_गाँठ(काष्ठा ggml_cgraph * cgraph, काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(cgraph->आकार > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

काष्ठा ggml_tensor * ggml_graph_get_tensor(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर अक्षर * name) {
    क्रम (पूर्णांक i = 0; i < cgraph->n_leafs; i++) {
        काष्ठा ggml_tensor * leaf = cgraph->leafs[i];

        अगर (म_भेद(leaf->name, name) == 0) {
            वापस leaf;
        }
    }

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        अगर (म_भेद(गाँठ->name, name) == 0) {
            वापस गाँठ;
        }
    }

    वापस शून्य;
}

काष्ठा ggml_tensor * ggml_graph_get_grad(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर काष्ठा ggml_tensor * गाँठ) {
    स्थिर माप_प्रकार igrad = ggml_hash_find(&cgraph->visited_hash_set, गाँठ);
    वापस igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads ? cgraph->grads[igrad] : शून्य;
}

काष्ठा ggml_tensor * ggml_graph_get_grad_acc(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर काष्ठा ggml_tensor * गाँठ) {
    स्थिर माप_प्रकार igrad = ggml_hash_find(&cgraph->visited_hash_set, गाँठ);
    वापस igrad != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs ? cgraph->grad_accs[igrad] : शून्य;
}

व्योम ggml_graph_print(स्थिर काष्ठा ggml_cgraph * cgraph) {
    GGML_LOG_INFO("=== GRAPH ===\n");

    GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                गाँठ->ne[0], गाँठ->ne[1], गाँठ->ne[2],
                ggml_op_name(गाँठ->op), (गाँठ->flags & GGML_TENSOR_FLAG_PARAM) ? "x" :
                      ggml_graph_get_grad(cgraph, गाँठ) ? "g" : " ");
    }

    GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    क्रम (पूर्णांक i = 0; i < cgraph->n_leafs; i++) {
        काष्ठा ggml_tensor * गाँठ = cgraph->leafs[i];

        GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                गाँठ->ne[0], गाँठ->ne[1],
                ggml_op_name(गाँठ->op),
                ggml_get_name(गाँठ));
    }

    GGML_LOG_INFO("========================================\n");
}

अटल पूर्णांक ggml_गाँठ_list_find_tensor(स्थिर काष्ठा ggml_cgraph * cgraph,
                                      स्थिर पूर्णांक *                idxs,
                                      पूर्णांक                        गिनती,
                                      स्थिर काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(cgraph && idxs);
    क्रम (पूर्णांक i = 0; i < गिनती; ++i) {
        स्थिर पूर्णांक गाँठ_idx = idxs[i];

        अगर (गाँठ_idx >= cgraph->n_nodes) {
            वापस -1;
        }
        अगर (cgraph->nodes[गाँठ_idx] == tensor) {
            वापस i;
        }
    }
    वापस -1;
}

अटल bool ggml_is_constant(स्थिर काष्ठा ggml_tensor * tensor) {
    वापस tensor->बफर != शून्य && ggml_backend_बफर_get_usage(tensor->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS && (tensor->flags & GGML_TENSOR_FLAG_PARAM) == 0;
}

bool ggml_can_fuse_subgraph_ext(स्थिर काष्ठा ggml_cgraph * cgraph,
                                स्थिर पूर्णांक *                गाँठ_idxs,
                                पूर्णांक                        गिनती,
                                स्थिर क्रमागत ggml_op *       ops,
                                स्थिर पूर्णांक *                outputs,
                                पूर्णांक                        अंक_outputs) {
    GGML_ASSERT(outputs && अंक_outputs > 0);

    क्रम (पूर्णांक i = 0; i < गिनती; ++i) {
        अगर (गाँठ_idxs[i] >= cgraph->n_nodes) {
            वापस false;
        }

        स्थिर काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idxs[i]];

        अगर (गाँठ->op != ops[i]) {
            वापस false;
        }

        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            वापस false;
        }

        अगर (ggml_गाँठ_list_find_tensor(cgraph, outputs, अंक_outputs, गाँठ) != -1) {
            जारी;
        }

        अगर (गाँठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
            वापस false;
        }

        पूर्णांक subgraph_uses = 0;
        क्रम (पूर्णांक j = i + 1; j < गिनती; ++j) {
            स्थिर काष्ठा ggml_tensor * other_गाँठ = cgraph->nodes[गाँठ_idxs[j]];
            क्रम (पूर्णांक src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                अगर (other_गाँठ->src[src_idx] == गाँठ) {
                    subgraph_uses++;
                }
            }
        }

        अगर (subgraph_uses != ggml_गाँठ_get_use_गिनती(cgraph, गाँठ_idxs[i])) {
            वापस false;
        }

        // if node is a view, check if the view_src and all its parent view_srcs are within the subgraph.
        // external view sources are allowed only for weight tensors, which are constant for this graph execution.
        काष्ठा ggml_tensor * view_src = गाँठ->view_src;
        जबतक (view_src) {
            अगर (ggml_गाँठ_list_find_tensor(cgraph, गाँठ_idxs, गिनती, view_src) == -1 && !ggml_is_constant(view_src)) {
                वापस false;
            }
            view_src = view_src->view_src;
        }
    }

    वापस true;
}

// check if node is part of the graph
अटल bool ggml_graph_find(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर काष्ठा ggml_tensor * गाँठ) {
    अगर (cgraph == शून्य) {
        वापस true;
    }

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        अगर (cgraph->nodes[i] == गाँठ) {
            वापस true;
        }
    }

    वापस false;
}

अटल काष्ठा ggml_tensor * ggml_graph_get_parent(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर काष्ठा ggml_tensor * गाँठ) {
    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        काष्ठा ggml_tensor * parent = cgraph->nodes[i];
        काष्ठा ggml_tensor * grad = ggml_graph_get_grad(cgraph, parent);

        अगर (grad == गाँठ) {
            वापस parent;
        }
    }

    वापस शून्य;
}

अटल व्योम ggml_graph_dump_dot_गाँठ_edge(खाता * fp, स्थिर काष्ठा ggml_cgraph * gb, काष्ठा ggml_tensor * गाँठ, काष्ठा ggml_tensor * parent, स्थिर अक्षर * label)  {
    काष्ठा ggml_tensor * gparent = ggml_graph_get_parent(gb, गाँठ);
    काष्ठा ggml_tensor * gparent0 = ggml_graph_get_parent(gb, parent);
    ख_लिखो(fp, "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (व्योम *) gparent0 : (व्योम *) parent,
            gparent ? (व्योम *) gparent : (व्योम *) गाँठ,
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

अटल व्योम ggml_graph_dump_dot_leaf_edge(खाता * fp, काष्ठा ggml_tensor * गाँठ, काष्ठा ggml_tensor * parent, स्थिर अक्षर * label)  {
    ख_लिखो(fp, "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
            (व्योम *) parent,
            (व्योम *) गाँठ,
            label);
}

व्योम ggml_graph_dump_dot(स्थिर काष्ठा ggml_cgraph * gb, स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर अक्षर * filename) {
    अक्षर color[16];

    खाता * fp = ggml_fopen(filename, "w");
    GGML_ASSERT(fp);

    ख_लिखो(fp, "digraph G {\n");
    ख_लिखो(fp, "  newrank = true;\n");
    ख_लिखो(fp, "  rankdir = TB;\n");

    क्रम (पूर्णांक i = 0; i < gb->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = gb->nodes[i];
        काष्ठा ggml_tensor * grad = ggml_graph_get_grad(gb, गाँठ);

        अगर (ggml_graph_get_parent(gb, गाँठ) != शून्य) {
            जारी;
        }

        अगर (गाँठ->flags & GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, माप(color), "yellow");
        } अन्यथा अगर (grad) {
            अगर (ggml_graph_find(cgraph, गाँठ)) {
                snprintf(color, माप(color), "green");
            } अन्यथा {
                snprintf(color, माप(color), "lightblue");
            }
        } अन्यथा {
            snprintf(color, माप(color), "white");
        }

        ख_लिखो(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (व्योम *) गाँठ, color);

        अगर (म_माप(गाँठ->name) > 0) {
            ख_लिखो(fp, "%s (%s)|", गाँठ->name, ggml_type_name(गाँठ->type));
        } अन्यथा {
            ख_लिखो(fp, "(%s)|", ggml_type_name(गाँठ->type));
        }

        अगर (ggml_is_matrix(गाँठ)) {
            ख_लिखो(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, गाँठ->ne[0], गाँठ->ne[1], ggml_op_प्रतीक(गाँठ->op));
        } अन्यथा {
            ख_लिखो(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, गाँठ->ne[0], गाँठ->ne[1], गाँठ->ne[2], ggml_op_प्रतीक(गाँठ->op));
        }

        अगर (grad) {
            ख_लिखो(fp, " | <g>%s\"; ]\n", ggml_op_प्रतीक(grad->op));
        } अन्यथा {
            ख_लिखो(fp, "\"; ]\n");
        }
    }

    क्रम (पूर्णांक i = 0; i < gb->n_leafs; i++) {
        काष्ठा ggml_tensor * गाँठ = gb->leafs[i];

        snprintf(color, माप(color), "pink");

        ख_लिखो(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (व्योम *) गाँठ, color);

        अगर (म_माप(गाँठ->name) > 0) {
            ख_लिखो(fp, "%s (%s)|", गाँठ->name, ggml_type_name(गाँठ->type));
        } अन्यथा {
            ख_लिखो(fp, "(%s)|", ggml_type_name(गाँठ->type));
        }

        ख_लिखो(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, गाँठ->ne[0], गाँठ->ne[1]);
        अगर (ggml_nelements(गाँठ) < 5 && गाँठ->तथ्य != शून्य) {
            ख_लिखो(fp, " | (");
            क्रम (पूर्णांक j = 0; j < ggml_nelements(गाँठ); j++) {
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
                    ख_लिखो(fp, "#");
                }
                अगर (j < ggml_nelements(गाँठ) - 1) {
                    ख_लिखो(fp, ", ");
                }
            }
            ख_लिखो(fp, ")");
        }
        ख_लिखो(fp, "\"; ]\n");
    }

    क्रम (पूर्णांक i = 0; i < gb->n_nodes; i++) {
        काष्ठा ggml_tensor * गाँठ = gb->nodes[i];

        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            अगर (गाँठ->src[j]) {
                अक्षर label[16];
                snprintf(label, माप(label), "src %d", j);
                ggml_graph_dump_dot_गाँठ_edge(fp, gb, गाँठ, गाँठ->src[j], label);
            }
        }
    }

    क्रम (पूर्णांक i = 0; i < gb->n_leafs; i++) {
        काष्ठा ggml_tensor * गाँठ = gb->leafs[i];

        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            अगर (गाँठ->src[j]) {
                अक्षर label[16];
                snprintf(label, माप(label), "src %d", j);
                ggml_graph_dump_dot_leaf_edge(fp, गाँठ, गाँठ->src[j], label);
            }
        }
    }

    ख_लिखो(fp, "}\n");

    ख_बंद(fp);

    GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

व्योम ggml_set_निवेश(काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_निवेश;
}

व्योम ggml_set_output(काष्ठा ggml_tensor * tensor) {
    क्रम (काष्ठा ggml_tensor * cur = tensor; cur != शून्य; cur = cur->view_src) {
        cur->flags |= GGML_TENSOR_FLAG_OUTPUT;
    }
}

व्योम ggml_set_param(काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    tensor->flags |= GGML_TENSOR_FLAG_PARAM;
}

व्योम ggml_set_loss(काष्ठा ggml_tensor * tensor) {
    GGML_ASSERT(ggml_is_scalar(tensor));
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    tensor->flags |= GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

व्योम ggml_quantize_init(क्रमागत ggml_type type) {
    ggml_critical_section_start();

    चयन (type) {
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); अवरोध;
        हाल GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); अवरोध;
        हाल GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); अवरोध;
        शेष: // nothing
            अवरोध;
    }

    ggml_critical_section_end();
}

व्योम ggml_quantize_free(व्योम) {
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

bool ggml_quantize_requires_imatrix(क्रमागत ggml_type type) {
    वापस
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;//   ||
        //type == GGML_TYPE_IQ1_M;
}

माप_प्रकार ggml_quantize_chunk(
        क्रमागत ggml_type   type,
           स्थिर भग्न * src,
                  व्योम * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           स्थिर भग्न * imatrix) {
    स्थिर int64_t n = nrows * n_per_row;

    अगर (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != शून्य);
    }

    GGML_ASSERT(start % type_traits[type].blck_आकार == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    स्थिर माप_प्रकार start_row = start / n_per_row;
    स्थिर माप_प्रकार row_आकार  = ggml_row_आकार(type, n_per_row);

    माप_प्रकार परिणाम = 0;

    चयन (type) {
        हाल GGML_TYPE_Q1_0:    परिणाम = quantize_q1_0   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q2_0:    परिणाम = quantize_q2_0   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q4_0:    परिणाम = quantize_q4_0   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q4_1:    परिणाम = quantize_q4_1   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q5_0:    परिणाम = quantize_q5_0   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q5_1:    परिणाम = quantize_q5_1   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q8_0:    परिणाम = quantize_q8_0   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_MXFP4:   परिणाम = quantize_mxfp4  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_NVFP4:   परिणाम = quantize_nvfp4  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q2_K:    परिणाम = quantize_q2_K   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q3_K:    परिणाम = quantize_q3_K   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q4_K:    परिणाम = quantize_q4_K   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q5_K:    परिणाम = quantize_q5_K   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_Q6_K:    परिणाम = quantize_q6_K   (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_TQ1_0:   परिणाम = quantize_tq1_0  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_TQ2_0:   परिणाम = quantize_tq2_0  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ2_XXS: परिणाम = quantize_iq2_xxs(src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ2_XS:  परिणाम = quantize_iq2_xs (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ3_XXS: परिणाम = quantize_iq3_xxs(src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ3_S:   परिणाम = quantize_iq3_s  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ2_S:   परिणाम = quantize_iq2_s  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ1_S:   परिणाम = quantize_iq1_s  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ1_M:   परिणाम = quantize_iq1_m  (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ4_NL:  परिणाम = quantize_iq4_nl (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_IQ4_XS:  परिणाम = quantize_iq4_xs (src + start, (अक्षर *) dst + start_row * row_आकार, nrows, n_per_row, imatrix); अवरोध;
        हाल GGML_TYPE_F16:
            {
                माप_प्रकार elemsize = माप(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                परिणाम = n * elemsize;
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                माप_प्रकार elemsize = माप(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                परिणाम = n * elemsize;
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                माप_प्रकार elemsize = माप(भग्न);
                परिणाम = n * elemsize;
                स_नकल((uint8_t *)dst + start * elemsize, src + start, परिणाम);
            } अवरोध;
        शेष:
            निश्चित(false);
    }

    GGML_ASSERT(परिणाम == nrows * row_आकार);

    वापस परिणाम;
}

////////////////////////////////////////////////////////////////////////////////

व्योम ggml_log_get(ggml_log_callback * log_callback, व्योम ** user_तथ्य) {
    *log_callback = g_logger_state.log_callback;
    *user_तथ्य    = g_logger_state.log_callback_user_तथ्य;
}

व्योम ggml_log_set(ggml_log_callback log_callback, व्योम * user_तथ्य) {
    g_logger_state.log_callback = log_callback ? log_callback : ggml_log_callback_default;
    g_logger_state.log_callback_user_तथ्य = user_तथ्य;
}

व्योम ggml_threadpool_params_init(काष्ठा ggml_threadpool_params * p, पूर्णांक n_threads) {
    p->n_threads  = n_threads;
    p->prio       = 0;     // default priority (usually means normal or inherited)
    p->poll       = 50;    // hybrid-polling enabled
    p->strict_cpu = false; // no strict placement (all threads share same cpumask)
    p->paused     = false; // threads are ready to go
    स_रखो(p->cpumask, 0, GGML_MAX_N_THREADS); // all-zero means use the default affinity (usually inherited)
}

काष्ठा ggml_threadpool_params ggml_threadpool_params_default(पूर्णांक n_threads) {
    काष्ठा ggml_threadpool_params p;
    ggml_threadpool_params_init(&p, n_threads);
    वापस p;
}

bool ggml_threadpool_params_match(स्थिर काष्ठा ggml_threadpool_params * p0, स्थिर काष्ठा ggml_threadpool_params * p1) {
    अगर (p0->n_threads  != p1->n_threads  ) वापस false;
    अगर (p0->prio       != p1->prio       ) वापस false;
    अगर (p0->poll       != p1->poll       ) वापस false;
    अगर (p0->strict_cpu != p1->strict_cpu ) वापस false;
    वापस स_भेद(p0->cpumask, p1->cpumask, GGML_MAX_N_THREADS) == 0;
}
