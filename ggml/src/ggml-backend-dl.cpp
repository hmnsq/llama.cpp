#ಅನ್ತರ್ಭಾವ "ggml-backend-dl.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32

dl_handle * dl_ತುಂಬು_library(ಸ್ಥಿರಾಂಕ fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_ಬಹುಲಕ = ಗಣ_Error_ಬಹುಲಕ(SEM_FAILCRITICALERRORS);
    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ | SEM_FAILCRITICALERRORS);

    HMODULE handle = ತುಂಬು_Library_W(path.wstring().c_str());

    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ);

    ಹಿಂತಿರುಗಿ handle;
}

ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    DWORD old_ಬಹುಲಕ = ಗಣ_Error_ಬಹುಲಕ(SEM_FAILCRITICALERRORS);
    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ | SEM_FAILCRITICALERRORS);

    ಶೂನ್ಯ * p = (ಶೂನ್ಯ *) Get_Proc_ವಿಳಾಸ(handle, name);

    ಗಣ_Error_ಬಹುಲಕ(old_ಬಹುಲಕ);

    ಹಿಂತಿರುಗಿ p;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dl_error() {
    ಹಿಂತಿರುಗಿ "";
}

#ಇಲ್ಲದಿದ್ದರೆ

dl_handle * dl_ತುಂಬು_library(ಸ್ಥಿರಾಂಕ fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    ಹಿಂತಿರುಗಿ handle;
}

ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಹಿಂತಿರುಗಿ dlsym(handle, name);
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dl_error() {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *rslt = dlerror();
    ಹಿಂತಿರುಗಿ rslt != ಶೂನ್ಯಸೂಚಕ ? rslt : "";
}

#ಆದರೆ_ಅಂತ್ಯ
