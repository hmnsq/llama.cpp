.खण्डः .text.init, "ax", @progbits
.वैश्विक _start
_start:
    # initialize वैश्विक pointer
.option निक्षिप
.option norelax
    la    gp, __global_pointer$
.option उद्धर
    # Firmware sets stack pointer before launch
    # bss not allowed, no init
    आह्वान  entry_point
    li    a2, 0  /* KERNEL_RETURN_SUCCESS (0) */
    mv    a1, a0
    li    a0, 8  /* SYSCALL_RETURN_FROM_KERNEL (8) */
    ecall
