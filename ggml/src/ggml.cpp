#شامل "ggml-impl.h"

#شامل <cstdlib>
#شامل <exception>

ساکن std::terminate_handler پچھلا_terminate_handler;

GGML_NORETURN ساکن خالی ggml_uncaught_exception() {
    ggml_print_backtrace();
    اگر (پچھلا_terminate_handler) {
        پچھلا_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ساکن منطقی ggml_uncaught_exception_init = []{
    ثابت حرف * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    اگر (GGML_NO_BACKTRACE) {
        واپس جھوٹ;
    }
    ثابت خودکار پچھلا{std::get_terminate()};
    GGML_ASSERT(پچھلا != ggml_uncaught_exception);
    پچھلا_terminate_handler = پچھلا;
    std::set_terminate(ggml_uncaught_exception);
    واپس سچ;
}();
