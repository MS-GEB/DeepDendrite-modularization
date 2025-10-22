NEURON {
    POINT_PROCESS ReLU_Syn_avg
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE lr_start, lr_end, lr_step
    RANGE v_in, grad_out, delta_w
    RANGE v_in_sum, v_in_avg
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
    v (mV)
    i (nA)

    has_stdp
    lr_step
    v_in
    v_in_sum
    v_in_avg
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
    lr_step = 0
    v_in_sum = 0
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
        lr_step = lr_step + 1
        v_in_sum = v_in_sum + v_in
        v_in_avg = v_in_sum / lr_step
        if (v_in_avg > 0) {
            delta_w = learning_rate * grad_out * v_in_avg
        }
        else {
            delta_w = 0
        }
    }
}
