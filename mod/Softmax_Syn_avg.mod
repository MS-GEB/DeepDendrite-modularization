NEURON {
    POINT_PROCESS Softmax_Syn_avg
    RANGE i
    ELECTRODE_CURRENT i
    RANGE lr_start, lr_end, lr_step
    RANGE v_out_0, v_out_1, v_out_2, v_out_3, v_out_4, v_out_5, v_out_6, v_out_7, v_out_8, v_out_9
    RANGE v_out_sum_0, v_out_sum_1, v_out_sum_2, v_out_sum_3, v_out_sum_4, v_out_sum_5, v_out_sum_6, v_out_sum_7, v_out_sum_8, v_out_sum_9
    RANGE v_out_avg_0, v_out_avg_1, v_out_avg_2, v_out_avg_3, v_out_avg_4, v_out_avg_5, v_out_avg_6, v_out_avg_7, v_out_avg_8, v_out_avg_9
    RANGE s_sum
    RANGE target_0, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9
    RANGE grad_out_0, grad_out_1, grad_out_2, grad_out_3, grad_out_4, grad_out_5, grad_out_6, grad_out_7, grad_out_8, grad_out_9
}

PARAMETER {
    lr_start (ms)
    lr_end (ms)

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
}

ASSIGNED {
    v (mV)
    i (nA)

    has_stdp
    lr_step

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

    v_out_sum_0
    v_out_sum_1
    v_out_sum_2
    v_out_sum_3
    v_out_sum_4
    v_out_sum_5
    v_out_sum_6
    v_out_sum_7
    v_out_sum_8
    v_out_sum_9

    v_out_avg_0
    v_out_avg_1
    v_out_avg_2
    v_out_avg_3
    v_out_avg_4
    v_out_avg_5
    v_out_avg_6
    v_out_avg_7
    v_out_avg_8
    v_out_avg_9

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
    has_stdp = 0
    lr_step = 0

    v_out_sum_0 = 0
    v_out_sum_1 = 0
    v_out_sum_2 = 0
    v_out_sum_3 = 0
    v_out_sum_4 = 0
    v_out_sum_5 = 0
    v_out_sum_6 = 0
    v_out_sum_7 = 0
    v_out_sum_8 = 0
    v_out_sum_9 = 0
}

BREAKPOINT {
    i = 0

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
        v_out_sum_0 = v_out_sum_0 + v_out_0
        v_out_sum_1 = v_out_sum_1 + v_out_1
        v_out_sum_2 = v_out_sum_2 + v_out_2
        v_out_sum_3 = v_out_sum_3 + v_out_3
        v_out_sum_4 = v_out_sum_4 + v_out_4
        v_out_sum_5 = v_out_sum_5 + v_out_5
        v_out_sum_6 = v_out_sum_6 + v_out_6
        v_out_sum_7 = v_out_sum_7 + v_out_7
        v_out_sum_8 = v_out_sum_8 + v_out_8
        v_out_sum_9 = v_out_sum_9 + v_out_9

        v_out_avg_0 = v_out_sum_0 / lr_step
        v_out_avg_1 = v_out_sum_1 / lr_step
        v_out_avg_2 = v_out_sum_2 / lr_step
        v_out_avg_3 = v_out_sum_3 / lr_step
        v_out_avg_4 = v_out_sum_4 / lr_step
        v_out_avg_5 = v_out_sum_5 / lr_step
        v_out_avg_6 = v_out_sum_6 / lr_step
        v_out_avg_7 = v_out_sum_7 / lr_step
        v_out_avg_8 = v_out_sum_8 / lr_step
        v_out_avg_9 = v_out_sum_9 / lr_step

        s_sum = exp(v_out_avg_0) + exp(v_out_avg_1) + exp(v_out_avg_2) + exp(v_out_avg_3) + exp(v_out_avg_4) + exp(v_out_avg_5) + exp(v_out_avg_6) + exp(v_out_avg_7) + exp(v_out_avg_8) + exp(v_out_avg_9)
        grad_out_0 = target_0 - exp(v_out_avg_0) / s_sum
        grad_out_1 = target_1 - exp(v_out_avg_1) / s_sum
        grad_out_2 = target_2 - exp(v_out_avg_2) / s_sum
        grad_out_3 = target_3 - exp(v_out_avg_3) / s_sum
        grad_out_4 = target_4 - exp(v_out_avg_4) / s_sum
        grad_out_5 = target_5 - exp(v_out_avg_5) / s_sum
        grad_out_6 = target_6 - exp(v_out_avg_6) / s_sum
        grad_out_7 = target_7 - exp(v_out_avg_7) / s_sum
        grad_out_8 = target_8 - exp(v_out_avg_8) / s_sum
        grad_out_9 = target_9 - exp(v_out_avg_9) / s_sum
    }
    else {
        grad_out_0 = 0
        grad_out_1 = 0
        grad_out_2 = 0
        grad_out_3 = 0
        grad_out_4 = 0
        grad_out_5 = 0
        grad_out_6 = 0
        grad_out_7 = 0
        grad_out_8 = 0
        grad_out_9 = 0
    }
}
