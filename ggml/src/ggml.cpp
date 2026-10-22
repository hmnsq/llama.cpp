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

স্থির যৌক্তিক ggml_uncaught_exception_init = []{
    ধ্রুবক অক্ষর * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    যদি (GGML_NO_BACKTRACE) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক auto পূর্ব{std::get_terminate()};
    GGML_ASSERT(পূর্ব != ggml_uncaught_exception);
    previous_terminate_handler = পূর্ব;
    std::set_terminate(ggml_uncaught_exception);
    ফেরত সত্য;
}();
