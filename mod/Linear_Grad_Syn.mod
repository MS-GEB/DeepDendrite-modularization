NEURON {
    POINT_PROCESS Linear_Grad_Syn
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE grad_out
    RANGE w
}

PARAMETER {
    w
    g = 0.01 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)

    grad_out
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

BREAKPOINT {
    i = g * w * grad_out
}
