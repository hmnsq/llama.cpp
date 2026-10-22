#ಸೂಚನೆ once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#ಅನ್ತರ್ಭಾವ <unistd.h>

#ಅನ್ತರ್ಭಾವ <cstdint>

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_PROTOCOL_MAJOR 0
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_PROTOCOL_MINOR 2

#ವ್ಯಾಖ್ಯಾನಿಸು APIR_HANDSHAKE_MAGIC 0xab1e

ಎಣಿಕೆ ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_ಉದ್ದಳತೆ = 3,
};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint64_t ApirCommandFlags;

ಎಣಿಕೆ Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ {
    APIR_ತುಂಬು_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_ತುಂಬು_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_ತುಂಬು_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_ತುಂಬು_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_ತುಂಬು_LIBRARY_ಚಿಹ್ನೆ_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ                = 6,
};

ಎಣಿಕೆ Apir_Forward_Return_ಸಂಕೇತ {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_ಗಮ್ಯ_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_ತಳ_ಸೂಚ್ಯಂಕ             = 4,
};

__attribute__((ಬಳಸದ_2)) ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_command_name(ApirCommandType type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_HANDSHAKE:
            ಹಿಂತಿರುಗಿ "HandShake";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_LOADLIBRARY:
            ಹಿಂತಿರುಗಿ "LoadLibrary";
        ಸ್ಥಿತಿ APIR_COMMAND_TYPE_FORWARD:
            ಹಿಂತಿರುಗಿ "Forward";
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ "unknown";
    }
}

__attribute__((ಬಳಸದ_2)) ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_ತುಂಬು_library_error(Apir_ತುಂಬು_Library_Return_ಸಂಕೇತ ಸಂಕೇತ_2) {
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_ತುಂಬು_LIBRARY_ERROR(ಸಂಕೇತ_name) \
    ಮಾಡು {                                   \
        ಆದರೆ (ಸಂಕೇತ_2 == ಸಂಕೇತ_name)             \
            ಹಿಂತಿರುಗಿ #code_name;             \
    } ಆಗ (0)

    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_SUCCESS);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_ALREADY_LOADED);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_ENV_VAR_MISSING);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_CANNOT_OPEN);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_ಚಿಹ್ನೆ_MISSING);
    APIR_ತುಂಬು_LIBRARY_ERROR(APIR_ತುಂಬು_LIBRARY_INIT_ತಳ_ಸೂಚ್ಯಂಕ);

    ಹಿಂತಿರುಗಿ "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು APIR_ತುಂಬು_LIBRARY_ERROR
}

__attribute__((ಬಳಸದ_2)) ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * apir_forward_error(Apir_Forward_Return_ಸಂಕೇತ ಸಂಕೇತ_2) {
#ವ್ಯಾಖ್ಯಾನಿಸು APIR_FORWARD_ERROR(ಸಂಕೇತ_name) \
    ಮಾಡು {                              \
        ಆದರೆ (ಸಂಕೇತ_2 == ಸಂಕೇತ_name)        \
            ಹಿಂತಿರುಗಿ #code_name;        \
    } ಆಗ (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_ಗಮ್ಯ_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_ತಳ_ಸೂಚ್ಯಂಕ);

    ಹಿಂತಿರುಗಿ "Unknown APIR_COMMAND_TYPE_FORWARD error";

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು APIR_FORWARD_ERROR
}
