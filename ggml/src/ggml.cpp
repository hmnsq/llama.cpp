#ضمن "ggml-impl.h"

#ضمن <cstdlib>
#ضمن <exception>

ساكن std::terminate_handler previous_terminate_handler;

GGML_NORETURN ساكن فراغ ggml_uncaught_exception() {
    ggml_بغ_backtrace();
    إذا (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

ساكن منطقي ggml_uncaught_exception_init = []{
    ثابت حرف * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    إذا (GGML_NO_BACKTRACE) {
        أرجع خطأ;
    }
    ثابت auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    أرجع true;
}();
