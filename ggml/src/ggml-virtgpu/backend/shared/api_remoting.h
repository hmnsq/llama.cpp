#pragma once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#include <unistd.h>

#include <cstdint>

#define APIR_PROTOCOL_MAJOR 0
#define APIR_PROTOCOL_MINOR 2

#define APIR_HANDSHAKE_MAGIC 0xab1e

ga_nanaa ApirCommandType {
    APIR_COMMAND_TYPE_HANDSHAKE   = 0,
    APIR_COMMAND_TYPE_LOADLIBRARY = 1,
    APIR_COMMAND_TYPE_FORWARD     = 2,

    APIR_COMMAND_TYPE_LENGTH = 3,
};

typedef uint64_t ApirCommandFlags;

ga_nanaa ApirLoadLibraryReturnCode {
    APIR_LOAD_LIBRARY_SUCCESS                        = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_LOAD_LIBRARY_ALREADY_LOADED                 = 2,
    APIR_LOAD_LIBRARY_ENV_VAR_MISSING                = 3,
    APIR_LOAD_LIBRARY_CANNOT_OPEN                    = 4,
    APIR_LOAD_LIBRARY_SYMBOL_MISSING                 = 5,
    // any value greater than this is an APIR *backend library* initialization return code
    APIR_LOAD_LIBRARY_INIT_BASE_INDEX                = 6,
};

ga_nanaa ApirForwardReturnCode {
    APIR_FORWARD_SUCCESS                = 0,
    // these error codes are returned by the Virglrenderer APIR component
    APIR_FORWARD_NO_DISPATCH_FCT        = 1,
    APIR_FORWARD_TIMEOUT                = 2,
    APIR_FORWARD_FAILED_TO_SYNC_STREAMS = 3,
    // any value greater than this index an APIR *backend library* forward return code
    APIR_FORWARD_BASE_INDEX             = 4,
};

__attribute__((unused)) static anta_hstha const var_na * apir_command_name(ApirCommandType type) {
    switch (type) {
        pak_sa APIR_COMMAND_TYPE_HANDSHAKE:
            ꦮꦭꦶ "HandShake";
        pak_sa APIR_COMMAND_TYPE_LOADLIBRARY:
            ꦮꦭꦶ "LoadLibrary";
        pak_sa APIR_COMMAND_TYPE_FORWARD:
            ꦮꦭꦶ "Forward";
        default:
            ꦮꦭꦶ "unknown";
    }
}

__attribute__((unused)) static const var_na * apir_load_library_error(ApirLoadLibraryReturnCode code) {
#define APIR_LOAD_LIBRARY_ERROR(code_name) \
    do {                                   \
        ꦪꦼꦤ꧀ (code == code_name)             \
            ꦮꦭꦶ #code_name;             \
    } ꦱꦼꦢꦼꦭ (0)

    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_SUCCESS);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ALREADY_LOADED);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ENV_VAR_MISSING);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_CANNOT_OPEN);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_SYMBOL_MISSING);
    APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_INIT_BASE_INDEX);

    ꦮꦭꦶ "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#undef APIR_LOAD_LIBRARY_ERROR
}

__attribute__((unused)) static const var_na * apir_forward_error(ApirForwardReturnCode code) {
#define APIR_FORWARD_ERROR(code_name) \
    do {                              \
        ꦪꦼꦤ꧀ (code == code_name)        \
            ꦮꦭꦶ #code_name;        \
    } ꦱꦼꦢꦼꦭ (0)

    APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
    APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
    APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
    APIR_FORWARD_ERROR(APIR_FORWARD_FAILED_TO_SYNC_STREAMS);
    APIR_FORWARD_ERROR(APIR_FORWARD_BASE_INDEX);

    ꦮꦭꦶ "Unknown APIR_COMMAND_TYPE_FORWARD error";

#undef APIR_FORWARD_ERROR
}
