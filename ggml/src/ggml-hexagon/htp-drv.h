#ಸೂಚನೆ once

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <AEEStdErr.h>
#ಅನ್ತರ್ಭಾವ <rpcmem.h>
#ಅನ್ತರ್ಭಾವ <remote.h>
#ಅನ್ತರ್ಭಾವ <dspqueue.h>

#ಆದರೆ defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) ಬಾಹ್ಯ
#    ಇಲ್ಲದಿದ್ದರೆ
#        define HTPDRV_API __declspec(dllimport) ಬಾಹ್ಯ
#    endif
#ಇಲ್ಲದಿದ್ದರೆ
#    define HTPDRV_API __attribute__ ((visibility ("default"))) ಬಾಹ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ DSP_ಸ್ಥಾನಾಂತರ
#    define DSP_ಸ್ಥಾನಾಂತರ 0x80000400
#ಆದರೆ_ಅಂತ್ಯ

/* Errno for connection reset by peer. */
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#ಆದರೆ_ಅಂತ್ಯ

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    ಇಲ್ಲದಿದ್ದರೆ
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        ಇಲ್ಲದಿದ್ದರೆ
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#ಆದರೆ_ಅಂತ್ಯ

/* Include windows specific header files. */
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#ಆದರೆ_ಅಂತ್ಯ

/* Includes and defines for all HLOS except windows */
#ಆದರೆ !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#ಆದರೆ_ಅಂತ್ಯ

/* Includes and defines for Hexagon and all HLOS except Windows. */
#ಆದರೆ !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(_WIN32)
#    pragma weak remote_system_request
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#ಇಲ್ಲದಿದ್ದರೆ
#     define DSPQUEUE_TIMEOUT 1000000
#ಆದರೆ_ಅಂತ್ಯ

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API ಪೂರ್ಣಾಂಕ htpdrv_init(ಶೂನ್ಯ);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(ಪೂರ್ಣಾಂಕ domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API ಪೂರ್ಣಾಂಕ htpdrv_get_arch(ಪೂರ್ಣಾಂಕ domain, ಪೂರ್ಣಾಂಕ * arch);

HTPDRV_API ಪೂರ್ಣಾಂಕ remote_system_request(system_req_payload * req);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
