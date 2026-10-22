#ಸೂಚನೆ once

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#   define WIN32_LEAN_AND_ಸರಾಸರಿ
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#ಇಲ್ಲದಿದ್ದರೆ
#    include <dlfcn.h>
#    include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <filesystem>

namespace fs = std::filesystem;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32

using dl_handle = std::remove_ಸೂಚಕ_t<HMODULE>;

ರಚನೆ dl_handle_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dl_handle * dl_ತುಂಬು_library(ಸ್ಥಿರಾಂಕ fs::ಮಾರ್ಗ & ಮಾರ್ಗ) {
    // suppress error dialogs for missing DLLs
    DWORD old_ಬಹುಲಕ = ಗಣ_Error_ಬಹುಲಕ(SEM_FAILCRITICALERRORS);
    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ | SEM_FAILCRITICALERRORS);

    HMODULE handle = ತುಂಬು_Library_W(ಮಾರ್ಗ.wstring().c_str());

    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ);

    ಹಿಂತಿರುಗಿ handle;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    DWORD old_ಬಹುಲಕ = ಗಣ_Error_ಬಹುಲಕ(SEM_FAILCRITICALERRORS);
    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ | SEM_FAILCRITICALERRORS);

    ಶೂನ್ಯ * p = (ಶೂನ್ಯ *) Get_Proc_ವಿಳಾಸ(handle, name);

    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ);

    ಹಿಂತಿರುಗಿ p;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dl_error() {
    ಹಿಂತಿರುಗಿ "";
}

#ಇಲ್ಲದಿದ್ದರೆ

using dl_handle = ಶೂನ್ಯ;

ರಚನೆ dl_handle_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಶೂನ್ಯ * handle) {
        dlclose(handle);
    }
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ dl_handle * dl_ತುಂಬು_library(ಸ್ಥಿರಾಂಕ fs::ಮಾರ್ಗ & ಮಾರ್ಗ) {
    dl_handle * handle = dlopen(ಮಾರ್ಗ.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ಹಿಂತಿರುಗಿ handle;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಹಿಂತಿರುಗಿ dlsym(handle, name);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dl_error() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *rslt = dlerror();
    ಹಿಂತಿರುಗಿ rslt != nullptr ? rslt : "";
}

#ಆದರೆ_ಅಂತ್ಯ
