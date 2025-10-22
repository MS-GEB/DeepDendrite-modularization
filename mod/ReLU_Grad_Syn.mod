NEURON {
    POINT_PROCESS ReLU_Grad_Syn
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE v_in, grad_out
    RANGE w
}

PARAMETER {
    w
    g = 0.01 (uS)
}

ASSIGNED {
    v (mV)
    i (nA)

    v_in
    grad_out
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

BREAKPOINT {
    if (v_in > 0) {
        i = g * w * grad_out
    }
    else {
        i = 0
    }
}
