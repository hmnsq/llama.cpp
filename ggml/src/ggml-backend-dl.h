#رهنمود once

#اگر_تعریف_شده _WIN32
#   define WIN32_LEAN_AND_میانگین
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#پایان_اگر
#ضمن <filesystem>

namespace fs = std::filesystem;

#اگر_تعریف_شده _WIN32

using dl_handle = std::remove_اشاره_گر_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ عملگر()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ عملگر()(فراغ * handle) {
        dlclose(handle);
    }
};

#پایان_اگر

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_بارگیری_library(ثابت fs::مسیر & مسیر);
فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

