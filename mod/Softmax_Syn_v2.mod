NEURON {
    POINT_PROCESS Softmax_Syn_v2
    RANGE i, g
    ELECTRODE_CURRENT i
    RANGE v_out_0, v_out_1, v_out_2, v_out_3, v_out_4, v_out_5, v_out_6, v_out_7, v_out_8, v_out_9
    RANGE s_sum
    RANGE target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9
    RANGE smooth_target_0, smooth_target_1, smooth_target_2, smooth_target_3, smooth_target_4, smooth_target_5, smooth_target_6, smooth_target_7, smooth_target_8, smooth_target_9
    RANGE grad_out_0, grad_out_1, grad_out_2, grad_out_3, grad_out_4, grad_out_5, grad_out_6, grad_out_7, grad_out_8, grad_out_9
    RANGE epsilon, nclasses, momentum
}

PARAMETER {
    g = 0.01 (uS)

    target_0 = 0
    target_1 = 0
    target_2 = 0
    target_3 = 0
    target_4 = 0
    target_5 = 0
    target_6 = 0
    target_7 = 0
    target_8 = 0
    target_9 = 0

    epsilon = 0.1
    nclasses = 10
    momentum = 0.9
}

ASSIGNED {
    v (mV)
    i (nA)

    v_out_0
    v_out_1
    v_out_2
    v_out_3
    v_out_4
    v_out_5
    v_out_6
    v_out_7
    v_out_8
    v_out_9

    smooth_target_0
    smooth_target_1
    smooth_target_2
    smooth_target_3
    smooth_target_4
    smooth_target_5
    smooth_target_6
    smooth_target_7
    smooth_target_8
    smooth_target_9

    s_sum
    
    grad_out_0
    grad_out_1
    grad_out_2
    grad_out_3
    grad_out_4
    grad_out_5
    grad_out_6
    grad_out_7
    grad_out_8
    grad_out_9
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

INITIAL {
    : label smoothing
    if (fabs(target_0 - 1) < 1e-5) {
        smooth_target_0 = 1 - epsilon
    }
    else {
        smooth_target_0 = epsilon / (nclasses - 1)
    }
    if (fabs(target_1 - 1) < 1e-5) {
        smooth_target_1 = 1 - epsilon
    }
    else {
        smooth_target_1 = epsilon / (nclasses - 1)
    }
    if (fabs(target_2 - 1) < 1e-5) {
        smooth_target_2 = 1 - epsilon
    }
    else {
        smooth_target_2 = epsilon / (nclasses - 1)
    }
    if (fabs(target_3 - 1) < 1e-5) {
        smooth_target_3 = 1 - epsilon
    }
    else {
        smooth_target_3 = epsilon / (nclasses - 1)
    }
    if (fabs(target_4 - 1) < 1e-5) {
        smooth_target_4 = 1 - epsilon
    }
    else {
        smooth_target_4 = epsilon / (nclasses - 1)
    }
    if (fabs(target_5 - 1) < 1e-5) {
        smooth_target_5 = 1 - epsilon
    }
    else {
        smooth_target_5 = epsilon / (nclasses - 1)
    }
    if (fabs(target_6 - 1) < 1e-5) {
        smooth_target_6 = 1 - epsilon
    }
    else {
        smooth_target_6 = epsilon / (nclasses - 1)
    }
    if (fabs(target_7 - 1) < 1e-5) {
        smooth_target_7 = 1 - epsilon
    }
    else {
        smooth_target_7 = epsilon / (nclasses - 1)
    }
    if (fabs(target_8 - 1) < 1e-5) {
        smooth_target_8 = 1 - epsilon
    }
    else {
        smooth_target_8 = epsilon / (nclasses - 1)
    }
    if (fabs(target_9 - 1) < 1e-5) {
        smooth_target_9 = 1 - epsilon
    }
    else {
        smooth_target_9 = epsilon / (nclasses - 1)
    }
}

BREAKPOINT {
    i = 0

    s_sum = exp(v_out_0) + exp(v_out_1) + exp(v_out_2) + exp(v_out_3) + exp(v_out_4) + exp(v_out_5) + exp(v_out_6) + exp(v_out_7) + exp(v_out_8) + exp(v_out_9)
    grad_out_0 = smooth_target_0 - exp(v_out_0) / s_sum
    grad_out_1 = smooth_target_1 - exp(v_out_1) / s_sum
    grad_out_2 = smooth_target_2 - exp(v_out_2) / s_sum
    grad_out_3 = smooth_target_3 - exp(v_out_3) / s_sum
    grad_out_4 = smooth_target_4 - exp(v_out_4) / s_sum
    grad_out_5 = smooth_target_5 - exp(v_out_5) / s_sum
    grad_out_6 = smooth_target_6 - exp(v_out_6) / s_sum
    grad_out_7 = smooth_target_7 - exp(v_out_7) / s_sum
    grad_out_8 = smooth_target_8 - exp(v_out_8) / s_sum
    grad_out_9 = smooth_target_9 - exp(v_out_9) / s_sum
}
