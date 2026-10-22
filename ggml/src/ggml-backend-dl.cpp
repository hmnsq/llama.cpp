#ضمن "ggml-backend-dl.h"

#جے_تعریف_شدہ _WIN32

dl_handle * dl_لدو_library(ثابت fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    HMODULE handle = لدو_Library_W(path.wstring().c_str());

    سیٹ_Error_منوال(old_منوال);

    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    DWORD old_منوال = سیٹ_Error_منوال(SEM_FAILCRITICALERRORS);
    سیٹ_Error_منوال(old_منوال | SEM_FAILCRITICALERRORS);

    فراغ * p = (فراغ *) Get_Proc_پتہ(handle, name);

    سیٹ_Error_منوال(old_منوال);

    أرجع p;
}

ثابت حرف * dl_error() {
    أرجع "";
}

#وإلا

dl_handle * dl_لدو_library(ثابت fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    أرجع handle;
}

فراغ * dl_get_sym(dl_handle * handle, ثابت حرف * name) {
    أرجع dlsym(handle, name);
}

ثابت حرف * dl_error() {
    ثابت حرف *rslt = dlerror();
    أرجع rslt != خالی_سوچک ? rslt : "";
}

#جے_انت
