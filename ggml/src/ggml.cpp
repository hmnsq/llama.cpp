#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <exception>

স্থির std::terminate_handler previous_terminate_handler;

GGML_NORETURN স্থির শূন্য ggml_uncaught_exception() {
    ggml_print_backtrace();
    যদি (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

স্থির তার্কিক ggml_uncaught_exception_init = []{
    ধ্রুৱ ৱর্ণ * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    যদি (GGML_NO_BACKTRACE) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    ঘূৰাই সঁচা;
}();
