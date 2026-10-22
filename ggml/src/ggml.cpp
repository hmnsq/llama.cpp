#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <exception>

ಸ್ಥಿರ std::terminate_handler ಹಿಂದಿನ_terminate_handler;

GGML_NORETURN ಸ್ಥಿರ ಶೂನ್ಯ ggml_uncaught_exception() {
    ggml_print_backtrace();
    ಆದರೆ (ಹಿಂದಿನ_terminate_handler) {
        ಹಿಂದಿನ_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_uncaught_exception_init = []{
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ಆದರೆ (GGML_NO_BACKTRACE) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    ಹಿಂದಿನ_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    ಹಿಂತಿರುಗಿ ನಿಜ;
}();
