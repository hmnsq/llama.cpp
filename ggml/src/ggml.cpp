#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <exception>

સ્થિર std::terminate_handler previous_terminate_handler;

GGML_NORETURN સ્થિર શૂન્ય ggml_uncaught_exception() {
    ggml_print_backtrace();
    જો (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

સ્થિર તાર્કિક ggml_uncaught_exception_init = []{
    અચલ અક્ષર * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    જો (GGML_NO_BACKTRACE) {
        પાછા ખોટું;
    }
    અચલ auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    પાછા સાચું;
}();
