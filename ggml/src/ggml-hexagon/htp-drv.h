#ہدایت once

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

#اگر_تعریف_شدہ _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#اگر_ختم

#شامل <AEEStdErr.h>
#شامل <rpcmem.h>
#شامل <remote.h>
#شامل <dspqueue.h>

#اگر defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) خارجی
#    ورنہ
#        define HTPDRV_API __declspec(dllimport) خارجی
#    endif
#ورنہ
#    define HTPDRV_API __attribute__ ((visibility ("default"))) خارجی
#اگر_ختم

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#اگر_غیر_تعریف_شدہ DSP_انحراف
#    define DSP_انحراف 0x80000400
#اگر_ختم

/* Errno for connection reset by peer. */
#اگر_غیر_تعریف_شدہ ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#اگر_ختم

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#اگر_غیر_تعریف_شدہ SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    ورنہ
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        ورنہ
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#اگر_ختم

/* Include windows specific header files. */
#اگر_تعریف_شدہ _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#اگر_ختم

/* Includes and defines for all HLOS except windows */
#اگر !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#اگر_ختم

/* Includes and defines for Hexagon and all HLOS except Windows. */
#اگر !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#اگر_ختم

#اگر !defined(_WIN32)
#    pragma weak remote_system_request
#اگر_ختم

#اگر_تعریف_شدہ _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#ورنہ
#     define DSPQUEUE_TIMEOUT 1000000
#اگر_ختم

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API صحیح htpdrv_init(خالی);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(صحیح domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API صحیح htpdrv_get_arch(صحیح domain, صحیح * arch);

HTPDRV_API صحیح remote_system_request(system_req_payload * req);

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم
