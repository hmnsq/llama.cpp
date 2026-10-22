#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <exception>

സ്ഥിതം std::terminate_handler previous_terminate_handler;

GGML_NORETURN സ്ഥിതം ശൂന്യം ggml_uncaught_exception() {
    ggml_print_backtrace();
    എങ്കിൽ (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

സ്ഥിതം തർക്കം ggml_uncaught_exception_init = []{
    സ്ഥിരം അക്ഷരം * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    എങ്കിൽ (GGML_NO_BACKTRACE) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    തിരികെ സത്യം;
}();
