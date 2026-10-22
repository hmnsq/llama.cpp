#ਹਦਾਇਤ once

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <AEEStdErr.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <rpcmem.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <remote.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <dspqueue.h>

#ਜੇ defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) ਬਾਹਰੀ
#    ਨਹੀਂਤਾਂ
#        define HTPDRV_API __declspec(dllimport) ਬਾਹਰੀ
#    endif
#ਨਹੀਂਤਾਂ
#    define HTPDRV_API __attribute__ ((visibility ("default"))) ਬਾਹਰੀ
#ਜੇ_ਅੰਤ

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ DSP_ਵਿੱਥ
#    define DSP_ਵਿੱਥ 0x80000400
#ਜੇ_ਅੰਤ

/* Errno for connection reset by peer. */
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#ਜੇ_ਅੰਤ

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    ਨਹੀਂਤਾਂ
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        ਨਹੀਂਤਾਂ
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#ਜੇ_ਅੰਤ

/* Include windows specific header files. */
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#ਜੇ_ਅੰਤ

/* Includes and defines for all HLOS except windows */
#ਜੇ !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#ਜੇ_ਅੰਤ

/* Includes and defines for Hexagon and all HLOS except Windows. */
#ਜੇ !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#ਜੇ_ਅੰਤ

#ਜੇ !defined(_WIN32)
#    pragma weak remote_system_request
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#ਨਹੀਂਤਾਂ
#     define DSPQUEUE_TIMEOUT 1000000
#ਜੇ_ਅੰਤ

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API ਪੂਰਨ htpdrv_init(ਖਾਲੀ);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(ਪੂਰਨ domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API ਪੂਰਨ htpdrv_get_arch(ਪੂਰਨ domain, ਪੂਰਨ * arch);

HTPDRV_API ਪੂਰਨ remote_system_request(system_req_payload * req);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ
