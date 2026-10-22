#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#નહીંતર
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#અન્તર્ભાવ <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

સંરચના dl_handle_deleter {
    શૂન્ય operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

સ્થિર અંતઃરેખ dl_handle * dl_load_library(અચલ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    પાછા handle;
}

સ્થિર અંતઃરેખ શૂન્ય * dl_get_sym(dl_handle * handle, અચલ અક્ષર * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    શૂન્ય * p = (શૂન્ય *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    પાછા p;
}

સ્થિર અંતઃરેખ અચલ અક્ષર * dl_error() {
    પાછા "";
}

#નહીંતર

using dl_handle = શૂન્ય;

સંરચના dl_handle_deleter {
    શૂન્ય operator()(શૂન્ય * handle) {
        dlclose(handle);
    }
};

સ્થિર અંતઃરેખ dl_handle * dl_load_library(અચલ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    પાછા handle;
}

સ્થિર અંતઃરેખ શૂન્ય * dl_get_sym(dl_handle * handle, અચલ અક્ષર * name) {
    પાછા dlsym(handle, name);
}

સ્થિર અંતઃરેખ અચલ અક્ષર * dl_error() {
    અચલ અક્ષર *rslt = dlerror();
    પાછા rslt != nullptr ? rslt : "";
}

#endif
