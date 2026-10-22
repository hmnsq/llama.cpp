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

ساكن مضمن dl_handle * dl_بارول_library(ثابت fs::لار & لار) {
    // suppress error dialogs for missing DLLs
    DWORD old_ډېرپېښ = سېټ_Error_ډېرپېښ(SEM_FAILCRITICALERRORS);
    سېټ_Error_ډېرپېښ(old_ډېرپېښ | SEM_FAILCRITICALERRORS);

    HMODULE handle = بارول_Library_W(لار.wstring().c_str());

    سېټ_Error_ډېرپېښ(old_ډېرپېښ);

    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_ډېرپېښ = سېټ_Error_ډېرپېښ(SEM_FAILCRITICALERRORS);
    سېټ_Error_ډېرپېښ(old_ډېرپېښ | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پته(handle, name);

    سېټ_Error_ډېرپېښ(old_ډېرپېښ);

    أرجع p;
}

ساكن مضمن ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ چلوونکی()(فراغ * handle) {
        dlclose(handle);
    }
};

ساكن مضمن dl_handle * dl_بارول_library(ثابت fs::لار & لار) {
    dl_handle * handle = dlopen(لار.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ساكن مضمن ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != nullptr ? rslt : "";
}

#د_که_پای
