'''
Defines a class, Neuron472299294, of neurons from Allen Brain Institute's model 472299294

A demo is available by running:

    python -i mosinit.py
'''
class PassiveL23PC:
    _counter = 0
    def __init__(self, name="PassiveL23PC", x=0, y=0, z=0):
        '''Instantiate PassiveL23PC.
        
        Parameters:
            x, y, z -- position offset
            
        Note: if name is not specified, PassiveL23PC_instance is used instead
        '''
                
        self._name = name + f"[{type(self)._counter}]"
        type(self)._counter += 1
        # self._name = name
        # load the morphology
        from load_swc import load_swc
        load_swc('Rorb-IRES2-Cre-D_Ai14_IVSCC_-168052.03.02.01_397999191_m.swc', self,
                 use_axon=False, xshift=x, yshift=y, zshift=z)

        # custom axon (works because dropping axon during import)
        from neuron import h
        self.axon = [h.Section(cell=self, name='axon[0]'),
                     h.Section(cell=self, name='axon[1]')]
        for sec in self.axon:
            sec.L = 30
            sec.diam = 1
            sec.nseg = 1
        self.axon[0].connect(self.soma[0](0.5))
        self.axon[1].connect(self.axon[0](1))
        self.all += self.axon

        self._insert_mechanisms()
        self._discretize_model()
        self._set_mechanism_parameters()
    
    def __str__(self):
        if hasattr(self, '_name'):
            return self._name
        else:
            return "PassiveL23PC_instance"
                
    def _insert_mechanisms(self):
        from neuron import h
        for sec in self.all:
            sec.insert("pas")
        # for mech in [u'CaDynamics', u'Ca_HVA', u'Ca_LVA', u'Ih', u'Im', u'K_P', u'K_T', u'Kv3_1', u'NaTs', u'Nap', u'SK']:
        #     self.soma[0].insert(mech)
    
    def _set_mechanism_parameters(self):
        from neuron import h
        for sec in self.all:
            sec.Ra = 167.07
            sec.e_pas = 0.  #-90.582359314
        for sec in self.apic:
            sec.cm = 2.17
            sec.g_pas = 7.76313565885e-05
        for sec in self.axon:
            sec.cm = 1.0
            sec.g_pas = 0.000551976131423
        for sec in self.dend:
            sec.cm = 2.17
            sec.g_pas = 1.08526630687e-05
        for sec in self.soma:
            sec.cm = 1.0
            # sec.ena = 53.0
            # sec.ek = -107.0
            # sec.gbar_Im = 1.23992e-05
            # sec.gbar_Ih = 0.000331224
            # sec.gbar_NaTs = 0.452286
            # sec.gbar_Nap = 0.00101181
            # sec.gbar_K_P = 0.0370931
            # sec.gbar_K_T = 0.000133211
            # sec.gbar_SK = 0.000150215
            # sec.gbar_Kv3_1 = 0.0857197
            # sec.gbar_Ca_HVA = 0.000509044
            # sec.gbar_Ca_LVA = 0.00616339
            # sec.gamma_CaDynamics = 0.000993274
            # sec.decay_CaDynamics = 845.244
            sec.g_pas = 0.000199625
    
    def _discretize_model(self):
        for sec in self.all:
            sec.nseg = 1 + 2 * int(sec.L / 40)

