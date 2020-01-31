### Water Resource Users Association Model


## Import packages
import numpy as np
import pandas as pd
import scipy.signal as signal
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
import sqlite3
import multiprocessing
import json
import sys
import time


## Define Classes
class Precipitation(object):
    '''The seasonal rainfall parameters for both the catchment and the community 
    
    Attributes: alph_prec, lmbd_prec, prdy_prec, rcrd_prec
    '''
    def __init__(self):
        
        # Insert placeholders for attributes
        self.alph_prec = None
        self.lmbd_prec = None 
        self.rcrd_prec = None
        
    def prec_sims(self):
        '''Simulate seasonal rainfall.'''
        
        # Define season length
        rsea_prec = INVL_SIMU * (TSEA_CLIM + BUFF_SIMU)
        dsea_prec = INVL_SIMU * int(TGRW_CROP - TSEA_CLIM)
        
        # Choose seasonal parameters
        alph_prec = np.random.gamma(1 / ARCV_CLIM**2, 
                                    ARMN_CLIM * ARCV_CLIM**2, 1)[0]
        lmbd_prec = np.random.gamma(1 / LRCV_CLIM**2, 
                                    LRMN_CLIM * LRCV_CLIM**2, 1)[0]
        prdy_prec = alph_prec * lmbd_prec  # m day-1
        
        # Simulate precipitation for rainy and dry seasons
        rrcd_prec = np.random.binomial(1, lmbd_prec / INVL_SIMU, 
                                       rsea_prec).astype(np.float)
        drcd_prec = np.random.binomial(1, LDMN_CLIM / INVL_SIMU, 
                                       dsea_prec).astype(np.float)
        ramt_prec = np.random.exponential(alph_prec, 
                                          len(rrcd_prec[rrcd_prec > 0]))
        damt_prec = np.random.exponential(ADMN_CLIM, 
                                          len(drcd_prec[drcd_prec > 0]))
        np.place(rrcd_prec, rrcd_prec == 1, ramt_prec)
        np.place(drcd_prec, drcd_prec == 1, damt_prec)
        rcrd_prec = np.concatenate((rrcd_prec, drcd_prec))
        
        # Save attributes
        self.alph_prec = alph_prec
        self.lmbd_prec = lmbd_prec
        self.prdy_prec = prdy_prec
        self.rcrd_prec = np.array(rcrd_prec, dtype = float)

        
class Network(object):
    '''The network structure of the entire catchment
    
    Attributes: nseg_ntwk, catc_ntwk
    '''
    def __init__(self):
       
        # Define season length and number of subcatchments
        nseg_ntwk = 2 * MGTD_NTWK - 1
        
        # Construct network structure
        styp_ntwk = np.zeros(nseg_ntwk, dtype = int)
        for segn_ntwk in np.arange(nseg_ntwk):
            if segn_ntwk == 0:
                styp_ntwk[segn_ntwk] = -1
            elif segn_ntwk == nseg_ntwk - 1:
                styp_ntwk[segn_ntwk] = 1
            else:
                nepa_ntwk = sum(styp_ntwk[styp_ntwk > 0])
                nipa_ntwk = abs(sum(styp_ntwk[styp_ntwk < 0]))
                pepa_ntwk = (nipa_ntwk - nepa_ntwk) * (MGTD_NTWK - nepa_ntwk) \
                            / ((nipa_ntwk - nepa_ntwk + 1) 
                            * (2 * MGTD_NTWK - 2 - nipa_ntwk - nepa_ntwk))
                styp_segn = np.random.binomial(1, pepa_ntwk, 1)
                if styp_segn == 0:
                    styp_ntwk[segn_ntwk] = -1
                else:
                    styp_ntwk[segn_ntwk] = styp_segn
                
        # Define downstream neighbor for each subcatchment
        dwst_ntwk = np.zeros(nseg_ntwk, dtype = int)
        for segn_ntwk in np.arange(nseg_ntwk):
            if segn_ntwk == 0:
                dwst_ntwk[segn_ntwk] = -1
            elif styp_ntwk[segn_ntwk - 1] == -1:
                dwst_ntwk[segn_ntwk] = segn_ntwk - 1
            elif styp_ntwk[segn_ntwk - 1] == 1 and styp_ntwk[segn_ntwk - 2] == -1:
                dwst_ntwk[segn_ntwk] = segn_ntwk - 2
            elif styp_ntwk[segn_ntwk - 1] == 1 and styp_ntwk[segn_ntwk - 2] == 1:
                ndws_segn = np.where(np.cumsum(np.flip(styp_ntwk[0:segn_ntwk])) 
                                     == 0)[0][0] + 1
                dwst_ntwk[segn_ntwk] = segn_ntwk - ndws_segn
            
        # Define edge connections and convert to graph
        edgs_ntwk = list(zip(dwst_ntwk[1:], np.arange(nseg_ntwk)[1:]))
        catc_ntwk = nx.DiGraph(edgs_ntwk)
        
        # Create subcatchment object for each network node
        for node_ntwk in catc_ntwk.nodes():
            snwk_subc = list(nx.dfs_preorder_nodes(catc_ntwk, node_ntwk))
            upst_subc = list(catc_ntwk._succ[node_ntwk].keys())
            subc = SubCatchment(node_ntwk, snwk_subc, upst_subc)
            catc_ntwk.nodes[node_ntwk]['subc_ntwk'] = subc
        
        # Update catchment areas
        tlar_ntwk = np.sum([catc_ntwk.nodes[node_ntwk]['subc_ntwk'].area_subc 
                            for node_ntwk in catc_ntwk.nodes()])
        fcar_ntwk = AREA_NTWK / tlar_ntwk
        for node_ntwk in catc_ntwk.nodes():
            subc = catc_ntwk.nodes[node_ntwk]['subc_ntwk']
            subc.area_subc = fcar_ntwk * subc.area_subc
            subc.lnth_subc = 1.4 * ((subc.area_subc / 1000**2) / 2.58999)**0.6 \
                             * 1.60934 * 1000  # m
            catc_ntwk.nodes[node_ntwk]['subc_ntwk'] = subc
        
        # Save attributes
        self.nseg_ntwk = nseg_ntwk
        self.catc_ntwk = catc_ntwk
        
    def smst_sims(self, prec):
        '''Simulates soil moisture dynamics in all subcatchments'''
        for node_ntwk in self.catc_ntwk.nodes():
            self.catc_ntwk.nodes[node_ntwk]['subc_ntwk'].smst_sims(prec)
    
    def flow_sims(self):
        '''Simulate river flow out of all subcatchments without irrigation'''
        
        # Sort by subcatchment magnitude
        mags_ntwk = [(sbid_subc, subc['subc_ntwk'].magn_subc) 
                     for sbid_subc, subc in self.catc_ntwk.nodes.items()]
        mags_ntwk.sort(key=lambda tup: tup[1])
        for sbid_subc in mags_ntwk:
            self.catc_ntwk.nodes[sbid_subc[0]]['subc_ntwk'].flow_sims(self)


class SubCatchment(object):
    '''Subcatchment characteristics, precipitation, soil moisture and river flow
    
    Attributes: sbid_subc, magn_subc, snwk_subc, upst_subc, cwid_subc, styp_subc, 
    area_subc, lnth_subc, abst_subc, prec_subc, smst_subc, rnof_subc, leak_subc, 
    flow_subc, sbfl_subc, prfl_subc, abfl_subc
    '''
    def __init__(self, node_ntwk, snwk_subc, upst_subc):
        self.sbid_subc = node_ntwk
        self.magn_subc = int((len(snwk_subc) + 1) / 2)
        self.snwk_subc = snwk_subc
        self.upst_subc = upst_subc
        self.cwid_subc = -9999
        if self.magn_subc < 2:
            exar_subc = np.random.gamma(1 / EACV_SUBC**2, 
                                   EAMN_SUBC * EACV_SUBC**2, 1)[0]  # m**2
            self.styp_subc = 'exterior'
            self.area_subc = exar_subc
            self.lnth_subc = None
        else:
            inar_subc = np.random.gamma(1 / IACV_SUBC**2, 
                                   IAMN_SUBC * IACV_SUBC**2, 1)[0]  # m**2
            self.styp_subc = 'interior'
            self.area_subc = inar_subc
            self.lnth_subc = None
        
        # Set abstractions to 0
        self.abst_subc = np.zeros(INVL_SIMU * (TGRW_CROP + BUFF_SIMU))
        
        # Insert placeholders for other attributes
        self.prec_subc = None
        self.smst_subc = None
        self.rnof_subc = None
        self.leak_subc = None
        self.sbfl_subc = None
        self.flow_subc = None
        self.prfl_subc = None
        self.abfl_subc = None
    
    def _auih_subc(self, tmes_seas):
        '''Internal function to calculate the fast flow distribution'''
        ctka_subc = 1 / (CTMU_SUBC * self.area_subc**0.38)  # days**-1
        apdf_subc = ctka_subc * np.exp(-ctka_subc * tmes_seas)
        return apdf_subc
    
    def _buih_subc(self, tmes_seas):
        '''Internal function to calculate the base flow distribution'''
        ctkb_subc = 1 / (CTMU_SUBC * self.area_subc**0.38) / 10  # days**-1
        bpdf_subc = ctkb_subc * np.exp(-ctkb_subc * tmes_seas)
        return bpdf_subc
    
    def _cuih_subc(self, tmes_seas):
        '''Internal function to calculate the channel flow distribution'''
        cpdf_subc = (self.lnth_subc / (np.sqrt(4 * np.pi * CTDC_SUBC) 
                                       * tmes_seas**(3 / 2))) \
                    * np.exp(-((self.lnth_subc - CTWC_SUBC * tmes_seas)**2) 
                             / (4 * CTDC_SUBC * tmes_seas))  # days**-1
        return cpdf_subc
    
    def smst_sims(self, prec):
        '''Simulates soil moisture dynamics in the subcatchment'''
        
        prec_subc = prec.rcrd_prec - ALDF_CLIM * (1 - (np.exp(-(self.magn_subc - 1))))
        prec_subc[prec_subc < 0.] = 0.

        # Set initial values
        psms_invl = SFLD_SOIL 
        smst_subc = []
        rnof_subc = []
        leak_subc = []
        
        # Loop through intervals
        for prec_invl in prec_subc: 
            
            # Add Rainfall
            smst_invl = psms_invl + prec_invl / (PORO_SOIL * ZRCT_CROP)
            
            # Remove Surface Runoff
            if smst_invl > 1.:
                rnof_invl = smst_invl - 1.
                smst_invl = 1.
            else: 
                rnof_invl = 0.
            
            # Remove Leakage
            if smst_invl > SFLD_SOIL:
                leak_invl = (KSAT_SOIL / INVL_SIMU) \
                            * (np.exp(BETA_SOIL * (smst_invl - SFLD_SOIL)) - 1.) \
                            / (np.exp(BETA_SOIL * (1. - SFLD_SOIL)) - 1.)
                smst_invl = smst_invl - leak_invl / (PORO_SOIL * ZRCT_CROP)
            else:
                leak_invl = 0.
            
            # Remove Evapotranspiration Losses
            if smst_invl > SSTR_SOIL:
                evtr_invl = ETMX_CLIM / INVL_SIMU
            elif smst_invl > SWLT_SOIL and smst_invl <= SSTR_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) + ((ETMX_CLIM - EWLT_CLIM) / INVL_SIMU) \
                            * (smst_invl - SWLT_SOIL) / (SSTR_SOIL - SWLT_SOIL)
            elif smst_invl > SHYG_SOIL and smst_invl <= SWLT_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) * (smst_invl - SHYG_SOIL) \
                            / (SWLT_SOIL - SHYG_SOIL)
            else:
                evtr_invl = 0.
            smst_invl = smst_invl - evtr_invl / (PORO_SOIL * ZRCT_CROP)
            
            # Save interval Values
            smst_subc.append(smst_invl)
            rnof_subc.append(rnof_invl)
            leak_subc.append(leak_invl)
            
            # Update iterative values
            psms_invl = smst_invl
        
        # Save attributes
        self.prec_subc = np.array(prec_subc, dtype = float)
        self.smst_subc = np.array(smst_subc, dtype = float)
        self.rnof_subc = np.array(rnof_subc, dtype = float)
        self.leak_subc = np.array(leak_subc, dtype = float)
            
    def flow_sims(self, ntwk):
        '''Simulate river flow out of the subcatchment without irrigation'''
        
        # Create internal data structures
        lgrw_seas = INVL_SIMU * (TGRW_CROP + BUFF_SIMU)
        invl_seas = 1 + np.arange(lgrw_seas)
        apfc_seas = 1 + np.arange(APFC_SUBC * lgrw_seas)
        
        # Create catchment and channel hydrographs
        auih_aprx = self._auih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        buih_aprx = self._buih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        cuih_aprx = self._cuih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        
        # Calculate fast and slow responses 
        rnof_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, self.rnof_subc)
        leak_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, self.leak_subc)
        arsp_aprx = self.area_subc * (rnof_aprx + (1 - BFPR_SUBC) * leak_aprx)
        brsp_aprx = self.area_subc * BFPR_SUBC * leak_aprx
        
        # Retrive flow from contributing catcments and reinterpolate
        if len(self.upst_subc) < 1:
            upfl_subc = np.zeros(lgrw_seas)
        else:
            nods_ntwk = dict(ntwk.catc_ntwk.nodes(data = 'subc_ntwk'))
            upfl_subc = reduce(np.add, [nods_ntwk[subc_upst].flow_subc 
                                        for subc_upst in self.upst_subc])
        upfl_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, upfl_subc)
        
        # Calculate catchment and total flow then sub in alternative minima
        aflw_aprx = signal.fftconvolve(arsp_aprx, auih_aprx)[0:len(apfc_seas)]
        bflw_aprx = signal.fftconvolve(brsp_aprx, buih_aprx)[0:len(apfc_seas)]
        sbfl_aprx = aflw_aprx + bflw_aprx
        cflw_aprx = sbfl_aprx + upfl_aprx
        #flow_aprx = cflw_aprx
        flow_aprx = signal.fftconvolve(cflw_aprx, cuih_aprx)[0:len(apfc_seas)]
        sbfl_subc = sbfl_aprx[np.arange(0, len(sbfl_aprx), APFC_SUBC)]
        flow_subc = flow_aprx[np.arange(0, len(flow_aprx), APFC_SUBC)]
        abfl_subc = np.array(flow_subc, dtype = float)
        flow_subc[flow_subc < FLMN_SUBC / INVL_SIMU] = FLMN_SUBC / INVL_SIMU
        abfl_subc[abfl_subc < FLMN_SUBC / INVL_SIMU] = FLMN_SUBC / INVL_SIMU
        
        # Save attributes
        self.sbfl_subc = np.array(sbfl_subc, dtype = float)
        self.flow_subc = np.array(flow_subc, dtype = float)
        self.abfl_subc = np.array(flow_subc, dtype = float)
    
    def flow_updt(self, ntwk, wrua):
        '''Update river flow after upstream abstractions'''
        
         # Create internal data structures
        lgrw_seas = INVL_SIMU * (TGRW_CROP + BUFF_SIMU)
        invl_seas = 1 + np.arange(lgrw_seas)
        apfc_seas = 1 + np.arange(APFC_SUBC * lgrw_seas)
        
        # Create catchment and channel hydrographs
        auih_aprx = self._auih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        buih_aprx = self._buih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        cuih_aprx = self._cuih_subc(apfc_seas / (APFC_SUBC * INVL_SIMU)) \
                    / (APFC_SUBC * INVL_SIMU)
        
        # Calculate fast and slow responses 
        rnof_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, self.rnof_subc)
        leak_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, self.leak_subc)
        arsp_aprx = self.area_subc * (rnof_aprx + (1 - BFPR_SUBC) * leak_aprx)
        brsp_aprx = self.area_subc * BFPR_SUBC * leak_aprx
        
        # Retrive flow from contributing catcments and reinterpolate
        if len(self.upst_subc) < 1:
            upab_subc = np.zeros(lgrw_seas)
        else:
            nods_ntwk = dict(ntwk.catc_ntwk.nodes(data ='subc_ntwk'))
            upab_subc = reduce(np.add, [nods_ntwk[subc_upst].abfl_subc 
                                        for subc_upst in self.upst_subc])
        upab_aprx = np.interp(apfc_seas / APFC_SUBC, invl_seas, upab_subc)
        
        # Calculate total flow then sub in alternative minima
        aflw_aprx = signal.fftconvolve(arsp_aprx, auih_aprx)[0:len(apfc_seas)]
        bflw_aprx = signal.fftconvolve(brsp_aprx, buih_aprx)[0:len(apfc_seas)]
        sbfl_aprx = aflw_aprx + bflw_aprx
        cabf_aprx = sbfl_aprx + upab_aprx
        #abfl_aprx = cabf_aprx
        abfl_aprx = signal.fftconvolve(cabf_aprx, cuih_aprx)[0:len(apfc_seas)]
        abfl_subc = abfl_aprx[np.arange(0, len(abfl_aprx), APFC_SUBC)]
        abfl_subc[abfl_subc < FLMN_SUBC / INVL_SIMU] = FLMN_SUBC / INVL_SIMU
        
        # Save attributes
        self.abfl_subc = np.array(abfl_subc, dtype = float)
        self.prfl_subc = np.array(abfl_subc, dtype = float)

    def flow_abst(self, ntwk, wrua):
        '''Removes abstracted water from flow'''
        
        # Save catchment abstraction amounts
        cwpr = wrua.cwpr_wrua[self.cwid_subc]
        self.abst_subc[(INVL_SIMU * BUFF_SIMU):] = np.array(cwpr.abst_cwpr)
        
        # Update flow
        abfl_subc = self.abfl_subc - self.abst_subc
        abfl_subc[abfl_subc < FLMN_SUBC / INVL_SIMU] = FLMN_SUBC / INVL_SIMU
        
        # Save attributes
        self.abfl_subc = np.array(abfl_subc, dtype = float)


class WaterResourceUsersAssoc(object):
    '''The characteristics and outcomes of the WRUA
    
    Attributes: cwpr_wrua, loca_wrua
    '''
    def __init__(self, ntwk):
        
        # Create CWPs and localities
        self.cwpr_wrua = {}
        self.loca_wrua = {}
        
        # Determine number of restricted CWPs
        prob = np.random.random()        
        dcml = FCRS_WRUA * NCWP_WRUA - int(FCRS_WRUA * NCWP_WRUA)
        if prob <= dcml:
            nres_wrua = int(np.ceil(FCRS_WRUA * NCWP_WRUA))
        else:
            nres_wrua = int(np.floor(FCRS_WRUA * NCWP_WRUA))
        nunr_wrua = int(NCWP_WRUA - nres_wrua)
        stcw_wrua = np.append(np.zeros(nres_wrua), np.ones(nunr_wrua))
        np.random.shuffle(stcw_wrua)
        sccw_wrua = np.random.choice(np.arange(2 * MGTD_NTWK - 1), 2 * MGTD_NTWK - 1, 
                                     replace = False)[:NCWP_WRUA]
        nmem_cwpr =  int(NMMN_CWPR * (np.random.beta(1 / (8 * NMCV_CWPR**2) - 0.5, 
                                                     1 / (8 * NMCV_CWPR**2) - 0.5, 
                                                     1)[0] 
                                      + 0.5))
        for cwid_cwpr in range(NCWP_WRUA):
            
            # Create CWP
            cwpr = CommunityWaterProject(nmem_cwpr)
            cwpr.cwid_cwpr = cwid_cwpr
            cwpr.sbid_cwpr = sccw_wrua[cwid_cwpr]
            ntwk.catc_ntwk.nodes[cwpr.sbid_cwpr]['subc_ntwk'].cwid_subc = cwid_cwpr
            if stcw_wrua[cwid_cwpr] < 1:
                cwpr.stus_cwpr = 'restricted'
            else:
                cwpr.stus_cwpr = 'unrestricted'
            
            # Create Locality with same attributes
            loca = Locality()
            loca.cwid_loca = cwid_cwpr
            loca.sbid_loca = sccw_wrua[cwid_cwpr]
            loca.stus_loca = 'locality'
            loca.nmem_loca = np.array(cwpr.nmem_cwpr)
            loca.hhar_loca = np.array(cwpr.hhar_cwpr)
            loca.kcrp_loca = np.array(cwpr.kcrp_cwpr)
            
            # Save both
            self.cwpr_wrua.update({cwid_cwpr:cwpr})
            self.loca_wrua.update({cwid_cwpr:loca})
    
    def cwpr_sims(self, ntwk):
        '''Simulate CWP irrigation'''
        
        # Sort by subcatchment magnitude
        mags_ntwk = [(sbid_subc, subc['subc_ntwk'].magn_subc) 
                     for sbid_subc, subc in ntwk.catc_ntwk.nodes.items()]
        mags_ntwk.sort(key=lambda tup: tup[1])
        
        # Loop through subcatchments
        for sbid_subc in mags_ntwk:
            ntwk.catc_ntwk.nodes[sbid_subc[0]]['subc_ntwk'].flow_updt(ntwk, self)
            cwid_subc = ntwk.catc_ntwk.nodes[sbid_subc[0]]['subc_ntwk'].cwid_subc
            if cwid_subc != -9999:
                self.loca_wrua[cwid_subc].smst_sims(ntwk)
                self.cwpr_wrua[cwid_subc].abav_calc(ntwk)
                self.cwpr_wrua[cwid_subc].smst_sims(self)
                ntwk.catc_ntwk.nodes[sbid_subc[0]]['subc_ntwk'].flow_abst(ntwk, self)
    
    def cwpr_rslt(self, ntwk):
        '''Simulate seasonal results for CWPs'''
        
        # Loop through CWPs
        for cwpr in self.cwpr_wrua.values():
                        
            # Calculate illegal abstraction and assess fine
            abil_cwpr = np.round(cwpr.abst_cwpr - cwpr.abal_cwpr, 7)
            abil_cwpr[abil_cwpr < 0] = 0.
            abds_cwpr = np.add.reduceat(cwpr.abst_cwpr, 
                                        np.arange(0, len(cwpr.abst_cwpr), INVL_SIMU))
            abdo_cwpr = np.round(abds_cwpr - ABMX_WRUA, 7) 
            abdo_cwpr[abdo_cwpr < 0] = 0.
            if (sum(abil_cwpr) or sum(abdo_cwpr)) > 0:
                cwpr.sanc_cwpr = SANC_WRUA
            else:
                cwpr.sanc_cwpr = 0.
            
            # Process results
            self.loca_wrua[cwpr.cwid_cwpr].rslt_calc()
            cwpr.rslt_calc()

            
class Locality(object):
    '''The characteristics and outcomes of the surrounding locality
    
    Attributes: cwid_loca, sbid_loca, stus_loca, nmem_loca, hhar_loca, 
    kcrp_loca, prec_loca, smst_loca, strs_loca, stat_loca
    '''
    
    def __init__(self):
        
        # Set static and initial attributes
        self.cwid_loca = None
        self.sbid_loca = None
        self.stus_loca = None
        self.nmem_loca = None
        self.hhar_loca = None
        self.kcrp_loca = None
        
        # Insert placeholders for other attributes
        self.prec_loca = None
        self.smst_loca = None
        self.strs_loca = None
        self.stat_loca = None
    
    def smst_sims(self, ntwk):
        '''Assign soil moisture to locality'''
        
        # Determine soil moisture time series
        subc = ntwk.catc_ntwk.nodes[self.sbid_loca]['subc_ntwk']
        if BYCT_WRUA == True:
            prec_loca = subc.prec_subc
            smst_loca = subc.smst_subc
        else:
            prec_loca = ntwk.catc_ntwk.nodes[0]['subc_ntwk'].prec_subc
            smst_loca = ntwk.catc_ntwk.nodes[0]['subc_ntwk'].smst_subc
        
        # Save attributes
        self.prec_loca = np.array(prec_loca, dtype = float)
        self.smst_loca = np.array(smst_loca, dtype = float)
    
    def rslt_calc(self):
        '''Calculate the seasonal results for the locality'''
        
        # Calculate Average Static Stress
        sstr_loca = (SSTR_SOIL - self.smst_loca[(INVL_SIMU * BUFF_SIMU):]) \
                    / (SSTR_SOIL - SWLT_SOIL)  # dim
        sstr_loca = sstr_loca[sstr_loca > 0.]
        sstr_loca[sstr_loca > 1.] = 1.
        if len(sstr_loca) > 0:
            mstr_loca = np.mean(sstr_loca**QPAR_CROP)  # dim
        else:
            mstr_loca = 0. # dim
        
        # Calculate Crossing Parameters
        indx_loca = np.where(self.smst_loca[(INVL_SIMU * BUFF_SIMU):] >= SSTR_SOIL)
        ccrs_loca = np.diff(np.append(0, np.append(indx_loca, INVL_SIMU 
                                                   * TGRW_CROP + 1))) - 1
        ccrs_loca = ccrs_loca[ccrs_loca > 0]
        ncrs_loca = len(ccrs_loca)  # dim
        if ncrs_loca > 0:
            mcrs_loca = np.mean(ccrs_loca) / INVL_SIMU  # days
        else:
            mcrs_loca = 0.
        
        # Calculate dynamic stress
        dstr_loca = ((mstr_loca * mcrs_loca) \
                    / (self.kcrp_loca * TGRW_CROP))**(ncrs_loca**-RPAR_CROP)
        dstr_loca[dstr_loca > 1.] = 1.
        
        # Calculate crop yields
        ycrp_loca = self.hhar_loca * YMAX_CROP * (1. - dstr_loca)  # kg
        
        # Calculate total income
        retr_loca = COST_CROP * ycrp_loca  # $
        ntin_loca = np.array(retr_loca, dtype = float)
        
        # Save attributes
        self.strs_loca = np.array([np.mean(mstr_loca), np.mean(ncrs_loca), 
                                   np.mean(mcrs_loca), np.mean(dstr_loca)], 
                                  dtype = float)
        self.stat_loca = np.array([np.mean(ntin_loca), 
                                   len(ntin_loca[ntin_loca > CMIN_CWPR]) / len(ntin_loca), 
                                   np.nan], 
                                  dtype = float)


class CommunityWaterProject(object):
    '''The characteristics and outcomes of each CWP
    
    Attributes: cwid_cwpr, sbid_cwpr, stus_cwpr, nmem_cwpr, hhar_cwpr, 
    kcrp_cwpr, hhst_cwpr, abal_cwpr, abav_cwpr, smst_cwpr, stor_cwpr, 
    irrg_cwpr, abst_cwpr, wbal_cwpr, strs_cwpr, rslt_cwpr, stat_cwpr
    '''

    def __init__(self, nmem_cwpr):
        
        # Set static and initial attributes
        self.cwid_cwpr = None
        self.sbid_cwpr = None
        self.stus_cwpr = None
        self.nmem_cwpr = nmem_cwpr
        self.hhar_cwpr = HAMN_CWPR * (np.random.beta(1 / (8 * HACV_CWPR**2) - 0.5, 
                                                     1 / (8 * HACV_CWPR**2) - 0.5, 
                                                     self.nmem_cwpr)
                                      + 0.5)
        self.kcrp_cwpr = KPMN_CROP * (np.random.beta(1 / (8 * KPCV_CROP**2) - 0.5,
                                                     1 / (8 * KPCV_CROP**2) - 0.5, 
                                                     self.nmem_cwpr)
                                      + 0.5)
        self.hhst_cwpr = HSMN_CWPR * (np.random.beta(1 / (8 * HSCV_CWPR**2) - 0.5, 
                                                     1 / (8 * HSCV_CWPR**2) - 0.5, 
                                                     self.nmem_cwpr)
                                      + 0.5)  # m**3
        self.hhst_cwpr = np.round(self.hhst_cwpr, 2)
        self.cwrd_cwpr = int(np.floor(RDYS_WRUA * 
                                      np.random.randint(0, 10000, 1)[0] / 10000)) # dim
        
        # Insert placeholders for other attributes
        self.abal_cwpr = None
        self.abav_cwpr = None
        self.smst_cwpr = None
        self.stor_cwpr = None
        self.irrg_cwpr = None
        self.abst_cwpr = None
        self.sanc_cwpr = None
        self.wbal_cwpr = None
        self.strs_cwpr = None
        self.rslt_cwpr = None
        self.stat_cwpr = None
    
    def abav_calc(self, ntwk):
        '''Determine the water available for abstraction each time step'''
        
        # Determine CWP capacity
        abcp_cwpr = np.full(INVL_SIMU * TGRW_CROP, self.nmem_cwpr * PFMX_CWPR / INVL_SIMU)
        
        # Determine flow allowed considering uptake fraction, flow minimum
        subc = ntwk.catc_ntwk.nodes[self.sbid_cwpr]['subc_ntwk']
        flow_cwpr = subc.prfl_subc[(INVL_SIMU * BUFF_SIMU):]
        abal_cwpr = ABFC_WRUA * flow_cwpr - max(FLMN_CWPR, FLMN_WRUA) / INVL_SIMU
        abal_cwpr[abal_cwpr < 0] = 0.
        
        # Differentiate between restricted and unrestricted
        if self.stus_cwpr == 'restricted':
            abav_cwpr = np.array(abal_cwpr)
        elif self.stus_cwpr == 'unrestricted':
            abav_cwpr = flow_cwpr - FLMN_CWPR / INVL_SIMU
        
        # Account for rotation schedule
        abdy_cwpr = np.zeros(RDYS_WRUA)
        abdy_cwpr[self.cwrd_cwpr] = 1.
        abdy_cwpr = np.resize(np.repeat(abdy_cwpr, INVL_SIMU), INVL_SIMU * TGRW_CROP)
        abav_cwpr = abdy_cwpr * abav_cwpr

        # Calculate water available for abstraction 
        abav_cwpr = np.minimum(abcp_cwpr, abav_cwpr)
        abav_cwpr[abav_cwpr < 0] = 0.
        
        # Save attributes
        self.abal_cwpr = np.array(abal_cwpr, dtype = float)
        self.abav_cwpr = np.array(abav_cwpr, dtype = float)
    
    def smst_sims(self, wrua):
        '''Simulates soil moisture dynamics for the average CWP member'''
        
        # Determine when pipe flow and precipitation is available
        loca = wrua.loca_wrua[self.cwid_cwpr]
        abav_cwpr = np.array(self.abav_cwpr, dtype = float)
        prec_cwpr = loca.prec_loca[(INVL_SIMU * BUFF_SIMU):]
        
        # Set initial values
        psms_invl = loca.smst_loca[(INVL_SIMU * BUFF_SIMU - 1)]
        psto_invl = 0.
        smst_cwpr = []
        stor_cwpr = []
        irrg_cwpr = []
        abst_cwpr = []
        
        # Loop through intervals
        for prec_invl, abav_invl in zip(prec_cwpr, abav_cwpr): 
            
            # Check if abstraction has exceeded daily limit
            absm_invl = sum(abst_cwpr[(len(abst_cwpr) - len(abst_cwpr) % INVL_SIMU):])
            abrm_invl = max(0, round(ABMX_WRUA - absm_invl, 7))  # m**3
            #if self.stus_cwpr == 'restricted':
            #    abal_invl = min(abrm_invl, abav_invl)
            #elif self.stus_cwpr == 'unrestricted':
            #    abal_invl = abav_invl
            abal_invl = abav_invl

            # Add Rainfall and set storage
            smst_invl = psms_invl + prec_invl / (PORO_SOIL * ZRLC_CROP)
            stor_invl = psto_invl
            
            # Remove Surface Runoff
            if smst_invl > 1.:
                rnof_invl = smst_invl - 1.
                smst_invl = 1.
            else: 
                rnof_invl = 0.
            
            # Remove Leakage
            if smst_invl > SFLD_SOIL:
                leak_invl = (KSAT_SOIL / INVL_SIMU) \
                            * (np.exp(BETA_SOIL * (smst_invl - SFLD_SOIL)) - 1.) \
                            / (np.exp(BETA_SOIL * (1. - SFLD_SOIL)) - 1.)
                smst_invl = smst_invl - leak_invl / (PORO_SOIL * ZRLC_CROP)
            else:
                leak_invl = 0.
            
            # Remove Evapotranspiration Losses
            if smst_invl > SSTR_SOIL:
                evtr_invl = ETMX_CLIM / INVL_SIMU
            elif smst_invl > SWLT_SOIL and smst_invl <= SSTR_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) + ((ETMX_CLIM - EWLT_CLIM) / INVL_SIMU) \
                            * (smst_invl - SWLT_SOIL) / (SSTR_SOIL - SWLT_SOIL)
            elif smst_invl > SHYG_SOIL and smst_invl <= SWLT_SOIL:
                evtr_invl = (EWLT_CLIM / INVL_SIMU) * (smst_invl - SHYG_SOIL) \
                            / (SWLT_SOIL - SHYG_SOIL)
            else:
                evtr_invl = 0.
            smst_invl = smst_invl - evtr_invl / (PORO_SOIL * ZRLC_CROP)
            
            # Compare demand to available water and storage
            if smst_invl < SMIN_IRRG:
                irdm_invl = (SMAX_IRRG - smst_invl) * (PORO_SOIL * ZRLC_CROP) * np.sum(self.hhar_cwpr)  # m**3
            else:
                irdm_invl = 0. * np.sum(self.hhar_cwpr) # m**3
            abir_invl = min(irdm_invl, abal_invl)  # m**3
            umdm_invl = max(0, round(irdm_invl - abir_invl, 7))  # m**3
            stdm_invl = max(0, round(np.sum(self.hhst_cwpr) + STOR_CWPR - stor_invl, 7))  # m**3
            abex_invl = max(0, round(abal_invl - abir_invl, 7))  # m**3
            abso_invl = min(stdm_invl, abex_invl)  # m**3
            stir_invl = min(umdm_invl, stor_invl)  # m**3
            irrg_invl = abir_invl + abir_invl  # m**3
            
            # Update soil moisture and storage
            smst_invl = smst_invl + irrg_invl / np.sum(self.hhar_cwpr) \
                        / (PORO_SOIL * ZRLC_CROP) 
            stor_invl = round(stor_invl + abso_invl - stir_invl, 7)
            
            # Calculate water used
            abst_invl = abir_invl + abso_invl  # m**3
            
            # Save interval Values
            smst_cwpr.append(smst_invl)
            stor_cwpr.append(stor_invl)
            abst_cwpr.append(abst_invl)
            irrg_cwpr.append(irrg_invl)
            
            # Update iterative values
            psms_invl = smst_invl
            psto_invl = stor_invl
        
        # Save attributes
        self.smst_cwpr = np.array(smst_cwpr, dtype = float)
        self.stor_cwpr = np.array(stor_cwpr, dtype = float)
        self.abst_cwpr = np.array(abst_cwpr, dtype = float)
        self.irrg_cwpr = np.array(irrg_cwpr, dtype = float)
        
    def rslt_calc(self):
        '''Calculate the seasonal results for each CWP'''
        
        # Calculate Water Balance
        wbal_cwpr = np.round(self.abst_cwpr - self.irrg_cwpr 
                             - np.diff(np.insert(self.stor_cwpr, 0, 0.)), 7)  # m**3
        
        # Calculate Average Static Stress
        sstr_cwpr = (SSTR_SOIL - self.smst_cwpr) / (SSTR_SOIL - SWLT_SOIL)  # dim
        sstr_cwpr = sstr_cwpr[sstr_cwpr > 0.]
        sstr_cwpr[sstr_cwpr > 1.] = 1.
        if len(sstr_cwpr) > 0:
            mstr_cwpr = np.mean(sstr_cwpr**QPAR_CROP)  # dim
        else:
            mstr_cwpr = 0. # dim
        
        # Calculate Crossing Parameters
        indx_cwpr = np.where(self.smst_cwpr >= SSTR_SOIL)
        ccrs_cwpr = np.diff(np.append(0, np.append(indx_cwpr, INVL_SIMU 
                                                   * TGRW_CROP + 1))) - 1
        ccrs_cwpr = ccrs_cwpr[ccrs_cwpr > 0]
        ncrs_cwpr = len(ccrs_cwpr)  # dim
        if ncrs_cwpr > 0:
            mcrs_cwpr = np.mean(ccrs_cwpr) / INVL_SIMU  # days
        else:
            mcrs_cwpr = 0.
        
        # Calculate dynamic stress
        dstr_cwpr = ((mstr_cwpr * mcrs_cwpr) \
                    / (self.kcrp_cwpr * TGRW_CROP))**(ncrs_cwpr**-RPAR_CROP)
        dstr_cwpr[dstr_cwpr > 1.] = 1.
        
        # Calculate crop yields
        ycrp_cwpr = self.hhar_cwpr * YMAX_CROP * (1. - dstr_cwpr)  # kg
        
        # Calculate total income
        retr_cwpr = COST_CROP * ycrp_cwpr  # $
        
        # Calculate CWP costs
        mcst_cwpr = CAMO_CWPR * self.nmem_cwpr * np.random.exponential(1., 1)
        wlim_cwpr = MBLM_CWPR * self.nmem_cwpr * TGRW_CROP
        if np.sum(self.abst_cwpr) <= wlim_cwpr:
            wcst_cwpr = np.sum(self.abst_cwpr) * COST_CWPR
        else:
            wcst_cwpr = COST_CWPR * (wlim_cwpr + (np.sum(self.abst_cwpr) - wlim_cwpr) * 1.5)
        scst_cwpr = self.sanc_cwpr
        cfee_cwpr = (mcst_cwpr + wcst_cwpr + scst_cwpr) / self.nmem_cwpr
        
        # Calculate amount paid to CWP and net income
        cpad_cwpr = np.array(retr_cwpr, dtype = float)
        cpad_cwpr[cpad_cwpr > cfee_cwpr] = cfee_cwpr
        cpad_cwpr[cpad_cwpr < 0.] = 0.
        ntin_cwpr = retr_cwpr - cpad_cwpr
        ntin_cwpr[ntin_cwpr < 0.] = 0.
        
        # Save attributes
        self.wbal_cwpr = np.array(wbal_cwpr, dtype = float)
        self.strs_cwpr = np.array([np.mean(mstr_cwpr), np.mean(ncrs_cwpr), 
                                   np.mean(mcrs_cwpr), np.mean(dstr_cwpr)], 
                                  dtype = float)
        self.rslt_cwpr = np.array([mcst_cwpr, wcst_cwpr, scst_cwpr, np.sum(cpad_cwpr)], 
                                  dtype = float)
        self.stat_cwpr = np.array([np.mean(ntin_cwpr), 
                                   len(ntin_cwpr[ntin_cwpr > CMIN_CWPR]) / len(ntin_cwpr), 
                                   np.sum(cpad_cwpr) / (mcst_cwpr + wcst_cwpr + scst_cwpr)], 
                                  dtype = float)


class Results(object):
    '''The simulation results to be saved for analysis
    
    Attributes: prec_rslt, subc_rslt, loca_rslt, flow_rslt, abfl_rslt, cwpr_rslt
    '''
    
    def __init__(self):
        
        # Define Attributes
        self.prec_rslt = None
        self.subc_rslt = None
        self.loca_rslt = None
        self.flow_rslt = None
        self.abfl_rslt = None
        self.cwpr_rslt = None
        
   
    def rslt_clct(self, prec, ntwk, wrua):
        '''Collects results from all objects'''
        
        # Collect precipitation data
        prec_rslt = pd.DataFrame({'alph': pd.Series(prec.alph_prec, dtype = float), 
                                  'lmbd': pd.Series(prec.lmbd_prec, dtype = float), 
                                  'prdy': pd.Series(prec.prdy_prec, dtype = float)})
        
        # Collect subcatchment data
        subc_rslt = []
        for node_ntwk in ntwk.catc_ntwk.nodes():
            subc = ntwk.catc_ntwk.nodes[node_ntwk]['subc_ntwk']
            sbtl_rslt = np.array([subc.sbid_subc, subc.magn_subc, subc.cwid_subc, 
                                  subc.styp_subc, subc.area_subc, subc.lnth_subc])
            subc_rslt.append(sbtl_rslt)
        subc_rslt = pd.DataFrame(subc_rslt, 
                                 columns = ['sbid', 'magn', 'cwid', 'styp', 'area', 
                                            'lnth'], dtype = float)
        subc_rslt['sbid'] = subc_rslt['sbid'].astype(int)
        subc_rslt['magn'] = subc_rslt['magn'].astype(int)
        subc_rslt['cwid'] = subc_rslt['cwid'].astype(int)
        subc_rslt['styp'] = subc_rslt['styp'].astype(str)
        
        # Collect location data
        loca_rslt = []
        for loca in wrua.loca_wrua.values():
            lcid_rslt = np.array([loca.cwid_loca, loca.sbid_loca, loca.stus_loca, 
                                  loca.nmem_loca])
            lctl_rslt = np.concatenate((lcid_rslt, np.array([0.]), loca.stat_loca))
            loca_rslt.append(lctl_rslt)
        loca_rslt = pd.DataFrame(loca_rslt, 
                                 columns = ['cwid', 'sbid', 'stus', 'nmem', 'abst', 'avin', 
                                            'fcsc', 'fcpd'], 
                                 dtype = float)
        loca_rslt['cwid'] = loca_rslt['cwid'].astype(int)
        loca_rslt['sbid'] = loca_rslt['sbid'].astype(int)
        loca_rslt['stus'] = loca_rslt['stus'].astype(str)
        loca_rslt['nmem'] = loca_rslt['nmem'].astype(int)
        
        # Collect natural flow data
        otfl_rslt = ntwk.catc_ntwk.nodes[0]['subc_ntwk'].flow_subc [(INVL_SIMU * BUFF_SIMU):]
        flow_rslt = pd.DataFrame({'invl': pd.Series(np.arange(INVL_SIMU * TGRW_CROP), 
                                                    dtype = int), 
                                  'flow': pd.Series(otfl_rslt, dtype = float)})
        
        # Collect abstracted flow data
        otaf_rslt = ntwk.catc_ntwk.nodes[0]['subc_ntwk'].abfl_subc [(INVL_SIMU * BUFF_SIMU):]
        abfl_rslt = pd.DataFrame({'invl': pd.Series(np.arange(INVL_SIMU * TGRW_CROP), 
                                                    dtype = int),  
                                  'abfl': pd.Series(otaf_rslt, dtype = float)})
        
        # Collect CWP data
        cwpr_rslt = []
        for cwpr in wrua.cwpr_wrua.values():
            cwid_rslt = np.array([cwpr.cwid_cwpr, cwpr.sbid_cwpr, cwpr.stus_cwpr, 
                                  cwpr.nmem_cwpr])
            cwtl_rslt = np.concatenate((cwid_rslt, np.array([np.sum(cwpr.abst_cwpr)]), cwpr.stat_cwpr))
            cwpr_rslt.append(cwtl_rslt)
        cwpr_rslt = pd.DataFrame(cwpr_rslt, 
                                 columns = ['cwid', 'sbid', 'stus', 'nmem', 'abst', 'avin', 
                                            'fcsc', 'fcpd'], 
                                 dtype = float)
        cwpr_rslt['cwid'] = cwpr_rslt['cwid'].astype(int)
        cwpr_rslt['sbid'] = cwpr_rslt['sbid'].astype(int)
        cwpr_rslt['stus'] = cwpr_rslt['stus'].astype(str)
        cwpr_rslt['nmem'] = cwpr_rslt['nmem'].astype(int)
                
        # Save attributes
        self.prec_rslt = prec_rslt
        self.subc_rslt = subc_rslt
        self.loca_rslt = loca_rslt
        self.flow_rslt = flow_rslt
        self.abfl_rslt = abfl_rslt
        self.cwpr_rslt = cwpr_rslt


# Define simulation function
def modl_simu(irun_simu):
    '''Run model and save results'''
    
    # Set new seed for each run
    np.random.seed(irun_simu)
    
    # Initialize objects
    prec = Precipitation()
    ntwk = Network()
    wrua = WaterResourceUsersAssoc(ntwk)
    rslt = Results()
    
    # Run model
    prec.prec_sims()
    ntwk.smst_sims(prec)
    ntwk.flow_sims()
    wrua.cwpr_sims(ntwk)
    wrua.cwpr_rslt(ntwk)
    rslt.rslt_clct(prec, ntwk, wrua)
    
    # Add new column with simulation number
    rslt.prec_rslt.insert(0, "irun", np.full(rslt.prec_rslt.shape[0], irun_simu), False)
    rslt.subc_rslt.insert(0, "irun", np.full(rslt.subc_rslt.shape[0], irun_simu), False)
    rslt.loca_rslt.insert(0, "irun", np.full(rslt.loca_rslt.shape[0], irun_simu), False)
    rslt.flow_rslt.insert(0, "irun", np.full(rslt.flow_rslt.shape[0], irun_simu), False)
    rslt.abfl_rslt.insert(0, "irun", np.full(rslt.abfl_rslt.shape[0], irun_simu), False)
    rslt.abfl_rslt.insert(0, "simu", np.full(rslt.abfl_rslt.shape[0], NMBR_SIMU), False)
    rslt.cwpr_rslt.insert(0, "irun", np.full(rslt.cwpr_rslt.shape[0], irun_simu), False)
    rslt.cwpr_rslt.insert(0, "simu", np.full(rslt.cwpr_rslt.shape[0], NMBR_SIMU), False)
    
    # Upload tables to Sqlite database
    try:
        lock.acquire(True)
        
        # Static results
        if NMBR_SIMU == 0:
            rslt.prec_rslt.to_sql('prec_rslt', con = conn, if_exists = 'append', index = False)
            rslt.subc_rslt.to_sql('subc_rslt', con = conn, if_exists = 'append', index = False)
            rslt.loca_rslt.to_sql('loca_rslt', con = conn, if_exists = 'append', index = False)
            #rslt.flow_rslt.to_sql('flow_rslt', con = conn, if_exists = 'append', index = False)
        
        # Dynamic results
        #rslt.abfl_rslt.to_sql('abfl_rslt', con = conn, if_exists = 'append', index = False)
        rslt.cwpr_rslt.to_sql('cwpr_rslt', con = conn, if_exists = 'append', index = False)
        
    finally:
        lock.release()


## Import parameters and run model
json_file = sys.argv[1]

with open(json_file, "r") as para_file:
    para_data = json.load(para_file)

cats = ['CLIM', 'NTWK', 'SUBC', 'CWPR', 'CROP', 'SOIL', 'WRUA', 
		'IRRG', 'SIMU']
for catg in cats:
    for vble, valu in para_data[catg].items():
        try:
            exec(vble + ' = eval(valu)')
        except:
            exec(vble + ' = valu')
DBSE_FILE = para_data['DBSE_FILE']

# Create database and pool connections
conn = sqlite3.connect(DBSE_FILE)
lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

# Collect simulation data and upload to database
simu_data = pd.DataFrame({'simu': pd.Series(NMBR_SIMU, dtype = int),
                          'smin': pd.Series(SMIN_IRRG, dtype = float),
                          'nmmn': pd.Series(NMMN_CWPR, dtype = int),
                          'rdys': pd.Series(RDYS_WRUA, dtype = int),
                          'abmx': pd.Series(ABMX_WRUA, dtype = float),
                          'flmn': pd.Series(FLMN_WRUA, dtype = float),
                          'abfc': pd.Series(ABFC_WRUA, dtype = float), 
                          'fcrs': pd.Series(FCRS_WRUA, dtype = float)})
simu_data.to_sql('simu_data', con = conn, if_exists = 'append', index = False)

# Print simulation number
print('Simulation:' + str(NMBR_SIMU))

# Run process
pool.map(modl_simu, range(0, NRUN_SIMU))

# Close connections
pool.close()
pool.join()
conn.close()