#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <exception>

স্থির std::terminate_handler previous_terminate_handler;

GGML_NORETURN স্থির শূন্য ggml_uncaught_exception() {
    ggml_print_backtrace();
    যদি (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

স্থির তার্কিক ggml_uncaught_exception_init = []{
    ধ্রুব বর্ণ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    যদি (GGML_NO_BACKTRACE) {
        প্রত্যাগম অসত্য;
    }
    ধ্রুব auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    প্রত্যাগম সত্য;
}();
