#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <exception>

स्थिर std::terminate_handler previous_terminate_handler;

GGML_NORETURN स्थिर शून्य ggml_uncaught_exception() {
    ggml_print_backtrace();
    सचे (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

स्थिर तार्किक ggml_uncaught_exception_init = []{
    ध्रुव वर्ण * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    सचे (GGML_NO_BACKTRACE) {
        पटिनिवत्त मिच्छा;
    }
    ध्रुव auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    पटिनिवत्त सच्च;
}();
