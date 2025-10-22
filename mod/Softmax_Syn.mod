NEURON {
    POINT_PROCESS Softmax_Syn
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE v_out, target, grad_out
    RANGE epsilon, nclasses, smooth_target, momentum
    RANGE loss_fn
}

PARAMETER {
    g = 0.01 (uS)
    target = 0
    epsilon = 0.1
    nclasses = 10
    momentum = 0.9
    loss_fn = 0     : 0 for cross-entropy, 1 for mse
}

STATE {
    ra_v_out
    ra_v
}

ASSIGNED {
    v (mV)
    i (nA)

    smooth_target
    
    v_out
    grad_out
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

INITIAL {
    : label smoothing
    if (fabs(target - 1) < 1e-5) {
        smooth_target = 1 - epsilon
    }
    else {
        smooth_target = epsilon / (nclasses - 1)
    }
    ra_v_out = 0
    ra_v = 0
}

BREAKPOINT {
    SOLVE states

    if (loss_fn == 0) {
        i = g * exp(v_out)

        if (exp(ra_v_out) > ra_v) {
            grad_out = smooth_target - 1
        }
        else {
            grad_out = smooth_target - exp(ra_v_out) / ra_v
        }
    }
    else {
        i = 0
        grad_out = 2 * (smooth_target - v_out) / nclasses
    }
    
}

PROCEDURE states() {
    ra_v_out = momentum * ra_v_out + (1 - momentum) * v_out
    ra_v = momentum * ra_v + (1 - momentum) * v
}
