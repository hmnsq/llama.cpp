#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <exception>

स्थिर std::terminate_handler previous_terminate_handler;

GGML_NORETURN स्थिर शून्य ggml_uncaught_exception() {
    ggml_print_backtrace();
    जर (previous_terminate_handler) {
        previous_terminate_handler();
    }
    पात(); // unreachable unless previous_terminate_handler was nullptr
}

स्थिर तार्किक ggml_uncaught_exception_init = []{
    स्थिरांक अक्षर * GGML_NO_BACKTRACE = दो_पर्या("GGML_NO_BACKTRACE");
    जर (GGML_NO_BACKTRACE) {
        परत असत्य;
    }
    स्थिरांक स्वतः prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    परत सत्य;
}();
