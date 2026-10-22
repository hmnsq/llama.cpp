#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <exception>

ਸਥਿਰ std::terminate_handler ਪਿਛਲਾ_terminate_handler;

GGML_NORETURN ਸਥਿਰ ਖਾਲੀ ggml_uncaught_exception() {
    ggml_print_backtrace();
    ਜੇ (ਪਿਛਲਾ_terminate_handler) {
        ਪਿਛਲਾ_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_uncaught_exception_init = []{
    ਅਚਲ ਅੱਖਰ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ਜੇ (GGML_NO_BACKTRACE) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ ਸਵੈਚਾਲਿਤ prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    ਪਿਛਲਾ_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    ਵਾਪਸ ਸੱਚ;
}();
