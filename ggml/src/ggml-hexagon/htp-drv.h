#సూచన once

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

#నిర్వచితమైతే _WIN32
#    pragma clang diagnostic ignored "-Wignored-attributes"
#అయితే_ముగింపు

#అన్తర్భావ <AEEStdErr.h>
#అన్తర్భావ <rpcmem.h>
#అన్తర్భావ <remote.h>
#అన్తర్భావ <dspqueue.h>

#ఒకవేళ defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) బాహ్య
#    లేకపోతే
#        define HTPDRV_API __declspec(dllimport) బాహ్య
#    endif
#లేకపోతే
#    define HTPDRV_API __attribute__ ((visibility ("default"))) బాహ్య
#అయితే_ముగింపు

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#నిర్వచితం_కాకపోతే DSP_స్థానభ్రంశం
#    define DSP_స్థానభ్రంశం 0x80000400
#అయితే_ముగింపు

/* Errno for connection reset by peer. */
#నిర్వచితం_కాకపోతే ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#అయితే_ముగింపు

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#నిర్వచితం_కాకపోతే SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    లేకపోతే
#        ifdef _WIN32
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        లేకపోతే
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#అయితే_ముగింపు

/* Include windows specific header files. */
#నిర్వచితమైతే _WIN32
#    include <windows.h>
#    include <sysinfoapi.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#అయితే_ముగింపు

/* Includes and defines for all HLOS except windows */
#ఒకవేళ !defined(__hexagon__) && !defined(_WIN32)
#    include "unistd.h"

#    include <sys/time.h>
#అయితే_ముగింపు

/* Includes and defines for Hexagon and all HLOS except Windows. */
#ఒకవేళ !defined(_WIN32)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#అయితే_ముగింపు

#ఒకవేళ !defined(_WIN32)
#    pragma weak remote_system_request
#అయితే_ముగింపు

#నిర్వచితమైతే _WIN32
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE
#లేకపోతే
#     define DSPQUEUE_TIMEOUT 1000000
#అయితే_ముగింపు

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API పూర్ణాంకం htpdrv_init(శూన్య);

/**
 * htpdrv_get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * htpdrv_get_domain(పూర్ణాంకం domain_id);

/**
 * htpdrv_get_arch API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API పూర్ణాంకం htpdrv_get_arch(పూర్ణాంకం domain, పూర్ణాంకం * arch);

HTPDRV_API పూర్ణాంకం remote_system_request(system_req_payload * req);

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు
