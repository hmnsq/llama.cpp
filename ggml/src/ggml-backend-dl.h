#لارښوونه once

#که_تعریف_شوی _WIN32
#   define WIN32_LEAN_AND_منځنۍ
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#د_که_پای
#ضمن <filesystem>

namespace fs = std::filesystem;

#که_تعریف_شوی _WIN32

using dl_handle = std::remove_اشاره_ګر_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ چلوونکی()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ چلوونکی()(فراغ * handle) {
        dlclose(handle);
    }
};

#د_که_پای

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_بارول_library(ثابت fs::لار & لار);
فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name);
ثابت حرف * dl_error();

