#شامل "ggml-impl.h"

#شامل <cstdlib>
#شامل <exception>

ایستا std::terminate_handler previous_terminate_handler;

GGML_NORETURN ایستا تهی ggml_uncaught_exception() {
    ggml_print_backtrace();
    اگر (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ایستا منطقی ggml_uncaught_exception_init = []{
    ثابت نویسه * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    اگر (GGML_NO_BACKTRACE) {
        بازگشت نادرست;
    }
    ثابت auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    بازگشت درست;
}();
