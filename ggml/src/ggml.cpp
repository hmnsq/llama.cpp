#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <exception>

ಸ್ಥಿರ std::terminate_handler previous_terminate_handler;

GGML_NORETURN ಸ್ಥಿರ ಶೂನ್ಯ ggml_uncaught_exception() {
    ggml_print_backtrace();
    ಯದಿ (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_uncaught_exception_init = []{
    ಧ್ರುವ ವರ್ಣ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ಯದಿ (GGML_NO_BACKTRACE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}();
