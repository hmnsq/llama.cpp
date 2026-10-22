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

ساكن مضمن dl_handle * dl_بارگیری_library(ثابت fs::مسیر & مسیر) {
    // suppress error dialogs for missing DLLs
    DWORD old_نما = مجموعه_Error_نما(SEM_FAILCRITICALERRORS);
    مجموعه_Error_نما(old_نما | SEM_FAILCRITICALERRORS);

    HMODULE handle = بارگیری_Library_W(مسیر.wstring().c_str());

    مجموعه_Error_نما(old_نما);

    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_نما = مجموعه_Error_نما(SEM_FAILCRITICALERRORS);
    مجموعه_Error_نما(old_نما | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_آدرس(handle, name);

    مجموعه_Error_نما(old_نما);

    أرجع p;
}

ساكن مضمن ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ عملگر()(فراغ * handle) {
        dlclose(handle);
    }
};

ساكن مضمن dl_handle * dl_بارگیری_library(ثابت fs::مسیر & مسیر) {
    dl_handle * handle = dlopen(مسیر.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ساكن مضمن ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != nullptr ? rslt : "";
}

#پایان_اگر
