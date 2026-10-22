#समावेश "ggml-impl.h"

#समावेश <cstdlib>
#समावेश <exception>

अटल std::terminate_handler previous_terminate_handler;

GGML_NORETURN अटल व्योम ggml_uncaught_exception() {
    ggml_print_backtrace();
    अगर (previous_terminate_handler) {
        previous_terminate_handler();
    }
    पात(); // unreachable unless previous_terminate_handler was nullptr
}

अटल bool ggml_uncaught_exception_init = []{
    स्थिर अक्षर * GGML_NO_BACKTRACE = दो_पर्या("GGML_NO_BACKTRACE");
    अगर (GGML_NO_BACKTRACE) {
        वापस false;
    }
    स्थिर स्वतः पूर्व{std::get_terminate()};
    GGML_ASSERT(पूर्व != ggml_uncaught_exception);
    previous_terminate_handler = पूर्व;
    std::set_terminate(ggml_uncaught_exception);
    वापस true;
}();
