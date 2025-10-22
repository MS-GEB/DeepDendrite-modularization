/*
Copyright (c) 2016, Blue Brain Project
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "coreneuron/nrnconf.h"
#include "coreneuron/nrnoc/multicore.h"
#include "coreneuron/nrnoc/nrnoc_decl.h"
#include "coreneuron/nrniv/profiler_interface.h"

namespace coreneuron {
void nrn_finitialize(int setv, double v) {
    int i;
    NrnThread* _nt;

    Instrumentor::phase_begin("finitialize");
    t = 0.;
    dt2thread(-1.);
    nrn_thread_table_check();
    clear_event_queue();
    nrn_spike_exchange_init();
#if VECTORIZE
    nrn_play_init(); /* Vector.play */
                     /// Play events should be executed before initializing events
    for (i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The play events at t=0 */
    }
    if (setv) {
        for (_nt = nrn_threads; _nt < nrn_threads + nrn_nthread; ++_nt) {
            double* vec_v = &(VEC_V(0));
// clang-format off
            #pragma acc parallel loop present(      \
                _nt[0:1], vec_v[0:_nt->end])        \
                if (_nt->compute_gpu)
            // clang-format on
            for (i = 0; i < _nt->end; ++i) {
                vec_v[i] = v;
            }
        }
    }

    if (set_reset_v)
    {
        for (_nt = nrn_threads; _nt < nrn_threads + nrn_nthread; ++_nt)
        {
            double *vec_v = &(VEC_V(0));
            double *reset_v = _nt->resetv_each_node;
            #pragma acc parallel loop present(_nt[0:1],  \
                vec_v[0:_nt->end], reset_v[0:_nt->end])  \
                if (_nt->compute_gpu)
            for (i = 0; i < _nt->end; i++)
            {
                vec_v[i] = reset_v[i];
            }
        }
    }

    if (nrn_have_gaps) {
        nrnmpi_v_transfer();
        for (i = 0; i < nrn_nthread; ++i) {
            nrnthread_v_transfer(nrn_threads + i);
        }
    }
    
    if (has_non_v_gap)
    {
        for (i = 0; i < nrn_nthread; ++i)
        {
            if (nrn_threads[i].ncell > 0)
                gap_non_v_transfer(nrn_threads + i);
        }
    }

    if (is_training || is_testing)
    {
        for (i = 0; i < nrn_nthread; ++i)
        {
            if (nrn_threads[i].ncell > 0)
            {
                NrnThread* nt = nrn_threads + i;
                nrn_threads[i].iinput_file = 0;
                nrn_threads[i].istim = 0;
                nrn_threads[i].adam_beta_1_t = 1.;
                nrn_threads[i].adam_beta_2_t = 1.;
                #pragma acc update device(nt->iinput_file) if (nt->compute_gpu) async(nt->stream_id)
                #pragma acc update device(nt->istim) if (nt->compute_gpu) async(nt->stream_id)
                #pragma acc update device(nt->lr_scale) if (nt->compute_gpu)
                #pragma acc update device(nt->lr_start) if (nt->compute_gpu)
                #pragma acc update device(nt->lr_end) if (nt->compute_gpu)
                #pragma acc update device(nt->adam_beta_1_t) if (nt->compute_gpu) async(nt->stream_id)
                #pragma acc update device(nt->adam_beta_2_t) if (nt->compute_gpu) async(nt->stream_id)
                #pragma acc wait(nt->stream_id)
                set_stim_and_labels(nrn_threads + i);
            }
        }
    }

    for (i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, BEFORE_INITIAL);
    }
    /* the INITIAL blocks are ordered so that mechanisms that write
       concentrations are after ions and before mechanisms that read
       concentrations.
    */
    /* the memblist list in NrnThread is already so ordered */
    for (i = 0; i < nrn_nthread; ++i) {
        NrnThread* nt = nrn_threads + i;
        NrnThreadMembList* tml;
        for (tml = nt->tml; tml; tml = tml->next) {
            mod_f_t s = memb_func[tml->index].initialize;
            if (s) {
                (*s)(nt, tml->ml, tml->index);
            }
        }
    }
#endif

    init_net_events();
    for (i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, AFTER_INITIAL);
    }
    for (i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The INITIAL sent events at t=0 */
    }
    for (i = 0; i < nrn_nthread; ++i) {
        setup_tree_matrix_minimal(nrn_threads + i);
    }
    for (i = 0; i < nrn_nthread; ++i) {
        nrn_ba(nrn_threads + i, BEFORE_STEP);
    }
    for (i = 0; i < nrn_nthread; ++i) {
        nrn_deliver_events(nrn_threads + i); /* The record events at t=0 */
    }
#if NRNMPI
    nrn_spike_exchange(nrn_threads);
#endif
    Instrumentor::phase_end("finitialize");
}
}  // namespace coreneuron
