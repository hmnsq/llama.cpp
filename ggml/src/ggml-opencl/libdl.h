#ہدایت once

#جے_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#جے_انت
#ضمن <filesystem>

namespace fs = std::filesystem;

#جے_تعریف_شدہ _WIN32

using dl_handle = std::remove_اشارہ_گر_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ عامل()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ساكن مضمن dl_handle * dl_لدو_library(ثابت fs::رستہ & رستہ) {
    // suppress error dialogs for missing DLLs
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    HMODULE handle = لدو_Library_W(رستہ.wstring().c_str());

    سیٹ_Error_منوال(old_منوال);

    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پتہ(handle, name);

    سیٹ_Error_منوال(old_منوال);

    أرجع p;
}

ساكن مضمن ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

using dl_handle = فراغ;

بنية dl_handle_deleter {
    فراغ عامل()(فراغ * handle) {
        dlclose(handle);
    }
};

ساكن مضمن dl_handle * dl_لدو_library(ثابت fs::رستہ & رستہ) {
    dl_handle * handle = dlopen(رستہ.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ساكن مضمن ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != nullptr ? rslt : "";
}

#جے_انت
