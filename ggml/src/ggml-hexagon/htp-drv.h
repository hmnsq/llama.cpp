#pragma once

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

#ifdef _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#endif

#অন্তর্ভাব <AEEStdErr.h>
#অন্তর্ভাব <rpcmem.h>
#অন্তর্ভাব <remote.h>
#অন্তর্ভাব <dspqueue.h>

#যদি defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) বাহ্যিক
#    অন্যথা
#        define HTPDRV_API __declspec(dllimport) বাহ্যিক
#    endif
#অন্যথা
#    define HTPDRV_API __attribute__ ((visibility ("default"))) বাহ্যিক
#endif

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#ifndef DSP_OFFSET
#    define DSP_OFFSET 0x80000400
#endif

/* Errno for connection reset by peer. */
#ifndef ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#endif

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#ifndef SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    অন্যথা
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        অন্যথা
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#endif

/* Include windows specific header files. */
#ifdef _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#endif

/* Includes and defines for all HLOS except windows */
#যদি !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#endif

/* Includes and defines for Hexagon and all HLOS except Windows. */
#যদি !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#endif

#যদি !defined(_WIN32)
#    pragma weak remote_system_request
#endif

#ifdef _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#অন্যথা
#     define DSPQUEUE_TIMEOUT 1000000
#endif

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API পূর্ণসংখ্যা htpdrv_init(শূন্য);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(পূর্ণসংখ্যা domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API পূর্ণসংখ্যা htpdrv_get_arch(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা * arch);

HTPDRV_API পূর্ণসংখ্যা remote_system_request(system_req_payload * req);

#ifdef __cplusplus
}
#endif
