#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <exception>

நிலையான std::terminate_handler previous_terminate_handler;

GGML_NORETURN நிலையான வெறுமை ggml_uncaught_exception() {
    ggml_print_backtrace();
    எனில் (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

நிலையான தருக்க ggml_uncaught_exception_init = []{
    மாறிலி எழுத்து * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    எனில் (GGML_NO_BACKTRACE) {
        திரும்பு பொய்;
    }
    மாறிலி auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    திரும்பு உண்மை;
}();
