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

#ifndef multicore_h
#define multicore_h

#include "coreneuron/nrnconf.h"
#include "coreneuron/nrnoc/membfunc.h"
#include <vector>
#include <string>
#include <stdio.h>

namespace coreneuron {
class NetCon;
class PreSyn;

/*
   Point_process._presyn, used only if its NET_RECEIVE sends a net_event, is
   eliminated. Needed only by net_event function. Replaced by
   PreSyn* = nt->presyns + nt->pnt2presyn_ix[pnttype2presyn[pnt->_type]][pnt->_i_instance];
*/
extern int nrn_has_net_event_cnt_; /* how many net_event sender types are there? */
extern int* nrn_has_net_event_;    /* the types that send a net_event */
extern int* pnttype2presyn; /* from the type, which array of pnt2presyn_ix are we talking about. */

struct NrnThreadMembList { /* patterned after CvMembList in cvodeobj.h */
    NrnThreadMembList* next;
    Memb_list* ml;
    int index;
    int* dependencies; /* list of mechanism types that this mechanism depends on*/
    int ndependencies; /* for scheduling we need to know the dependency count */
};

struct NrnThreadBAList {
    Memb_list* ml; /* an item in the NrnThreadMembList */
    BAMech* bam;
    NrnThreadBAList* next;
};

/* for OpenACC, in order to avoid an error while update PreSyn, with virtual base
 * class, we are adding helper with flag variable which could be updated on GPU
 */
struct PreSynHelper {
    int flag_;
};


struct NrnThread {
    double _t;
    double _dt;
    double cj;
    double dt_io;


    NrnThreadMembList* tml;
    Memb_list** _ml_list;
    Point_process* pntprocs;  // synapses and artificial cells with and without gid
    PreSyn* presyns;          // all the output PreSyn with and without gid
    PreSynHelper* presyns_helper;
    int** pnt2presyn_ix;  // eliminates Point_process._presyn used only by net_event sender.
    NetCon* netcons;
    double* weights;  // size n_weight. NetCon.weight_ points into this array.

    int n_pntproc, n_presyn, n_input_presyn, n_netcon, n_weight;  // only for model_size

    int ncell; /* analogous to old rootnodecount */
    int end;   /* 1 + position of last in v_node array. Now v_node_count. */
    int id;    /* this is nrn_threads[id] */
    int _stop_stepping;
    int n_vecplay; /* number of instances of VecPlayContinuous */

    size_t _ndata, _nidata, _nvdata; /* sizes */
    double* _data;                   /* all the other double* and Datum to doubles point into here*/
    int* _idata;                     /* all the Datum to ints index into here */
    void** _vdata;                   /* all the Datum to pointers index into here */
    void** _vecplay;                 /* array of instances of VecPlayContinuous */

    double* _actual_rhs;
    double* _actual_d;
    double* _actual_a;
    double* _actual_b;
    double* _actual_v;
    double* _actual_area;
    double* _actual_diam; /* NULL if no mechanism has dparam with diam semantics */
    double* _shadow_rhs;  /* Not pointer into _data. Avoid race for multiple POINT_PROCESS in same
                             compartment */
    double* _shadow_d;    /* Not pointer into _data. Avoid race for multiple POINT_PROCESS in same
                             compartment */
    int* _v_parent_index;
    int* _permute;
    char* _sp13mat;              /* handle to general sparse matrix */
    Memb_list* _ecell_memb_list; /* normally nil */

    double _ctime; /* computation time in seconds (using nrnmpi_wtime) */

    NrnThreadBAList* tbl[BEFORE_AFTER_SIZE]; /* wasteful since almost all empty */

    int shadow_rhs_cnt; /* added to facilitate the NrnThread transfer to GPU */
    int compute_gpu;    /* define whether to compute with gpus */
    int stream_id;      /* define where the kernel will be launched on GPU stream */
    int _net_send_buffer_size;
    int _net_send_buffer_cnt;
    int* _net_send_buffer;

    int* _watch_types; /* NULL or 0 terminated array of integers */
    void* mapping;     /* section to segment mapping information */

    FILE *fp;
    FILE *fp_rec;
    int nrec_param;
    size_t rec_len, irec, rec_stride_len, rec_last_len;
    int* rec_mech_types;
    int* rec_param_index;
    double** rec_ptrs;
    float* rec_vals;
    float* rec_times;
    bool* is_rec_v;

    int nnon_v_gap;
    int *gap_src_mech_types, *gap_tgt_mech_types; // gap junction for non-voltage parameters
    int *gap_src_param_idx, *gap_tgt_param_idx;
    double **gap_src_param_ptrs, **gap_tgt_param_ptrs;
    bool *gap_src_is_v, *gap_tgt_is_v;

    //for training
    float sim_time_per_img, passed_time;
    int hd2out_len, in2hd_len, stim_len;
    int ninput, nhidden, nout, nproj;
    int *hd2out_mech_types, *hd2out_idx, *in2hd_mech_types, *in2hd_idx;
    int *stim_mech_types, *stim_idx;
    double **in2hd_ptrs, **hd2out_ptrs, **stim_ptrs;
    int iinput_file;
    std::vector<std::string> *input_filenames;
    double *imgs;
    int *labels, istim, iter, epoch;
    int nimg, img_size;
    int batch_size, nweights, nweights_per_net, ndelta_weights;
    int *w_idx, *dw_idx, *w_mech_types, *dw_mech_types;
    int nlabel_params;
    int *label_mech_types, *label_idx;
    double **label_ptrs;
    double **w_ptrs, **dw_ptrs;
    int buffer_size;
    double *weights_buffer;
    double *grads_buffer;
    double adam_beta_1, adam_beta_2, adam_epsilon;
    double *adam_m, *adam_v;
    double adam_beta_1_t, adam_beta_2_t;
    double *adam_m_hat, *adam_v_hat;
    int in2hd_g_len;
    int *in2hd_g_mech_types, *in2hd_g_idx;
    double **in2hd_g_ptrs;
    int* w_buffer_idx; // index of weights in weights_buffer, used in shared_weight
    int* dw_buffer_idx; // index of dw in weights_buffer, used in shared_weight
    char rec_filename[300];

    double *resetv_each_cell, *resetv_each_node;
    double _actual_t;

    int rec_type;
    double mean_s_time, mean_e_time;
    double lr_scale;
    double lr_start, lr_end;
};

extern void nrn_threads_create(int n);
extern int nrn_nthread;
extern NrnThread* nrn_threads;
extern void nrn_multithread_job(void* (*)(NrnThread*));
extern void nrn_thread_table_check(void);

extern void nrn_threads_free(void);

extern int _nrn_skip_initmodel;

}  // namespace coreneuron

#endif
