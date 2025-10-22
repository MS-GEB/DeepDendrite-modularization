NEURON {
    POINT_PROCESS ReLU_Grad_Syn_avg
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE v_in, grad_out
    RANGE v_in_sum
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
    v_in_sum
    grad_out
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

INITIAL {
    v_in_sum = 0
}

BREAKPOINT {
    v_in_sum = v_in_sum + v_in
    if (v_in_sum > 0) {
        i = g * w * grad_out
    }
    else {
        i = 0
    }
}
