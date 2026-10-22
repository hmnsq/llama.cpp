#ہدایت once

#اگر_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ورنہ
#    include <dlfcn.h>
#    include <unistd.h>
#اگر_ختم
#شامل <filesystem>

namespace fs = std::filesystem;

#اگر_تعریف_شدہ _WIN32

using dl_handle = std::remove_اشارہ_گر_t<HMODULE>;

ساخت dl_handle_deleter {
    خالی عامل()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ساکن اندرونی dl_handle * dl_بھرو_library(ثابت fs::راستہ & راستہ) {
    // suppress error dialogs for missing DLLs
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    HMODULE handle = بھرو_Library_W(راستہ.wstring().c_str());

    سیٹ_Error_منوال(old_منوال);

    واپس handle;
}

ساکن اندرونی خالی * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    خالی * p = (خالی *) Get_Proc_پتہ(handle, name);

    سیٹ_Error_منوال(old_منوال);

    واپس p;
}

ساکن اندرونی ثابت حرف * dl_error() {
    واپس "";
}

#ورنہ

using dl_handle = خالی;

ساخت dl_handle_deleter {
    خالی عامل()(خالی * handle) {
        dlclose(handle);
    }
};

ساکن اندرونی dl_handle * dl_بھرو_library(ثابت fs::راستہ & راستہ) {
    dl_handle * handle = dlopen(راستہ.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    واپس handle;
}

ساکن اندرونی خالی * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    واپس dlsym(handle, name);
}

ساکن اندرونی ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    واپس rslt != nullptr ? rslt : "";
}

#اگر_ختم
