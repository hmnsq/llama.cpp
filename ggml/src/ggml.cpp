#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdlib>
#అన్తర్భావ <exception>

స్థిర std::terminate_handler మునుపటి_terminate_handler;

GGML_NORETURN స్థిర శూన్య ggml_uncaught_exception() {
    ggml_print_backtrace();
    ఒకవేళ (మునుపటి_terminate_handler) {
        మునుపటి_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

స్థిర తార్కిక ggml_uncaught_exception_init = []{
    స్థిరరాశి అక్షరం * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    ఒకవేళ (GGML_NO_BACKTRACE) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి స్వయంచాలక prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    మునుపటి_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    తిరిగి నిజం;
}();
