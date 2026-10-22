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

#ಇಲ್ಲದಿದ್ದರೆ

using dl_handle = ಶೂನ್ಯ;

ರಚನೆ dl_handle_deleter {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಶೂನ್ಯ * handle) {
        dlclose(handle);
    }
};

#ಆದರೆ_ಅಂತ್ಯ

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_ತುಂಬು_library(ಸ್ಥಿರಾಂಕ fs::ಮಾರ್ಗ & ಮಾರ್ಗ);
ಶೂನ್ಯ * dl_get_sym(dl_handle * handle, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dl_error();

