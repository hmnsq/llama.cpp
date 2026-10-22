#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <exception>

ସ୍ଥିର std::terminate_handler previous_terminate_handler;

GGML_NORETURN ସ୍ଥିର ଶୂନ୍ୟ ggml_uncaught_exception() {
    ggml_print_backtrace();
    ଯଦି (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ସ୍ଥିର ତାର୍କିକ ggml_uncaught_exception_init = []{
    ଅଚଳ ଅକ୍ଷର * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ଯଦି (GGML_NO_BACKTRACE) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    ଫେରନ୍ତୁ ସତ୍ୟ;
}();
