#ہدایت once

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

#جے_تعریف_شدہ _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#جے_انت

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
#جے_انت

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#جے_بنا_تعریف DSP_وتھ
#    define DSP_وتھ 0x80000400
#جے_انت

/* Errno for connection reset by peer. */
#جے_بنا_تعریف ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#جے_انت

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#جے_بنا_تعریف SLEEP
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
#جے_انت

/* Include windows specific header files. */
#جے_تعریف_شدہ _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#جے_انت

/* Includes and defines for all HLOS except windows */
#إذا !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#جے_انت

/* Includes and defines for Hexagon and all HLOS except Windows. */
#إذا !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#جے_انت

#إذا !defined(_WIN32)
#    pragma weak remote_system_request
#جے_انت

#جے_تعریف_شدہ _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#وإلا
#     define DSPQUEUE_TIMEOUT 1000000
#جے_انت

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

#جے_تعریف_شدہ __cplusplus
}
#جے_انت
