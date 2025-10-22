NEURON {
    POINT_PROCESS ReLU_Syn
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE lr_start, lr_end
    RANGE v_in, grad_out, delta_w
    RANGE w, learning_rate
}

PARAMETER {
    lr_start (ms)
    lr_end (ms)

    w
    learning_rate = 0.01
    g = 0.01 (uS)
}

ASSIGNED {
    has_stdp
    v (mV)
    i (nA)

    v_in
    grad_out
    delta_w
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

INITIAL {
    has_stdp = 0
    delta_w = 0
}

BREAKPOINT {
    if (v_in > 0) {
        i = g * w * v_in
    }
    else {
        i = 0
    }

    if (t < lr_start) {
        has_stdp = 0
    }
    else if (t > lr_end) {
        has_stdp = 0
    }
    else {
        has_stdp = 1
    }
    if (has_stdp == 1) {
        if (v_in > 0) {
            delta_w = delta_w + learning_rate * grad_out * v_in
        }
    }
}
