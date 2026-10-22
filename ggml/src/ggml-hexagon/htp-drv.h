#هدايت once

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#جيڪڏهن_پورو

#ضمن <AEEStdErr.h>
#ضمن <rpcmem.h>
#ضمن <remote.h>
#ضمن <dspqueue.h>

#إذا defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) خارجي
#    وإلا
#        define HTPDRV_API __declspec(dllimport) خارجي
#    endif
#وإلا
#    define HTPDRV_API __attribute__ ((visibility ("default"))) خارجي
#جيڪڏهن_پورو

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#جيڪڏهن_وصف_نه_ٿيل DSP_وٿي
#    define DSP_وٿي 0x80000400
#جيڪڏهن_پورو

/* Errno for connection reset by peer. */
#جيڪڏهن_وصف_نه_ٿيل ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#جيڪڏهن_پورو

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#جيڪڏهن_وصف_نه_ٿيل SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    وإلا
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        وإلا
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#جيڪڏهن_پورو

/* Include windows specific header files. */
#جيڪڏهن_وصف_ٿيل _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#جيڪڏهن_پورو

/* Includes and defines for all HLOS except windows */
#إذا !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#جيڪڏهن_پورو

/* Includes and defines for Hexagon and all HLOS except Windows. */
#إذا !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#جيڪڏهن_پورو

#إذا !defined(_WIN32)
#    pragma weak remote_system_request
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#وإلا
#     define DSPQUEUE_TIMEOUT 1000000
#جيڪڏهن_پورو

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API صحيح htpdrv_init(فراغ);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(صحيح domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API صحيح htpdrv_get_arch(صحيح domain, صحيح * arch);

HTPDRV_API صحيح remote_system_request(system_req_payload * req);

#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو
