#هدايت once

#جيڪڏهن_وصف_ٿيل _WIN32
#   define WIN32_LEAN_AND_سراسري
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#جيڪڏهن_پورو
#ضمن <filesystem>

namespace fs = std::filesystem;

#جيڪڏهن_وصف_ٿيل _WIN32

using dl_handle = std::remove_اشارو_t<HMODULE>;

بنية dl_handle_deleter {
    فراغ عامل()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ساكن مضمن dl_handle * dl_ڀريو_library(ثابت fs::رستو & رستو) {
    // suppress error dialogs for missing DLLs
    DWORD old_گهڻائي = سيٽ_Error_گهڻائي(SEM_FAILCRITICALERRORS);
    سيٽ_Error_گهڻائي(old_گهڻائي | SEM_FAILCRITICALERRORS);

    HMODULE handle = ڀريو_Library_W(رستو.wstring().c_str());

    سيٽ_Error_گهڻائي(old_گهڻائي);

    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_گهڻائي = سيٽ_Error_گهڻائي(SEM_FAILCRITICALERRORS);
    سيٽ_Error_گهڻائي(old_گهڻائي | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پتو(handle, name);

    سيٽ_Error_گهڻائي(old_گهڻائي);

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

ساكن مضمن dl_handle * dl_ڀريو_library(ثابت fs::رستو & رستو) {
    dl_handle * handle = dlopen(رستو.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

ساكن مضمن فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ساكن مضمن ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != nullptr ? rslt : "";
}

#جيڪڏهن_پورو
