#!/miniconda3/envs/py3_env/bin/python


### Create json parameter files for simulations


## Import Modules
import numpy as np
import itertools
import json


## Create template parameter dictionary
para_temp = {'CLIM': {'ARMN_CLIM': 0.012, 'ARCV_CLIM': 0.20, 'ADMN_CLIM': 0.012, 
                      'ALDF_CLIM': 0.009, 'LRMN_CLIM': 0.5, 'LRCV_CLIM': 0.20, 
                      'LDMN_CLIM': 0.10, 'ETMX_CLIM': 0.005, 'EWLT_CLIM': 0.0001, 
                      'TSEA_CLIM': 110}, 
             'NTWK': {'MGTD_NTWK': 8, 'AREA_NTWK': '15 * (1000**2)'},
             'SUBC': {'SAMN_SUBC': 'AREA_NTWK / (2 * MGTD_NTWK - 1)', 'EIRT_SUBC': 1.3, 
                      'EAMN_SUBC': 'SAMN_SUBC * (2 * MGTD_NTWK - 1)/ (MGTD_NTWK + (MGTD_NTWK - 1) / EIRT_SUBC)', 
                      'EACV_SUBC': 0.20, 'IAMN_SUBC': 'EAMN_SUBC / EIRT_SUBC', 
                      'IACV_SUBC': 0.20, 'CTMU_SUBC': '24 / 24 / 1000**0.76', 
                      'CTUV_SUBC': '0.75 * 60 * 60 * 24', 'CTWC_SUBC': '(3 / 2) * CTUV_SUBC', 
                      'CTDC_SUBC': '1000 * 60 * 60 * 24', 'BFPR_SUBC': 0.5, 
                      'APFC_SUBC': 100, 'FLMN_SUBC': '0.00001 * 60 * 60 * 24'}, 
             'SOIL': {'KSAT_SOIL': 0.8, 'PORO_SOIL': 0.43, 'BETA_SOIL': 13.8, 
                      'SHYG_SOIL': 0.14, 'SWLT_SOIL': 0.18, 'SSTR_SOIL': 0.46, 
                      'SFLD_SOIL': 0.56}, 
             'CROP': {'ZRCT_CROP': 0.5, 'ZRLC_CROP': 0.5, 'YMAX_CROP': 0.3, 
                      'QPAR_CROP': 2, 'KPMN_CROP': 0.5, 'KPCV_CROP': 0.20, 
                      'RPAR_CROP': 0.5, 'COST_CROP': 0.2, 'TGRW_CROP': 110}, 
             'CWPR': {'NMMN_CWPR': 'NA', 'NMCV_CWPR': 0.20, 'HAMN_CWPR': 5000, 
                      'HACV_CWPR': 0.20, 'HSMN_CWPR': 5, 'HSCV_CWPR': 0.20, 
                      'CMIN_CWPR': 200, 'CAMO_CWPR': 10, 'PFMX_CWPR': 40, 
                      'MBLM_CWPR': 3, 'COST_CWPR': 0.01, 'STOR_CWPR': 50,
                      'FLMN_CWPR': '0.0001 * 60 * 60 * 24'}, 
             'WRUA': {'NCWP_WRUA': 'int(2 * MGTD_NTWK - 1)', 'BYCT_WRUA': 'False', 'FCRS_WRUA': 'NA', 
                      'RDYS_WRUA': 'NA', 'ABMX_WRUA': 'NA', 'FLMN_WRUA': 'NA', 
                      'ABFC_WRUA': 'NA', 'SANC_WRUA': 0}, 
             'IRRG': {'SMIN_IRRG': 'NA', 'SMAX_IRRG': 'SFLD_SOIL'},
             'SIMU': {'NMBR_SIMU': 'NA', 'NRUN_SIMU': 10000, 'BUFF_SIMU' : 'int(0.3 * TGRW_CROP)', 
                      'INVL_SIMU': 24},
             'DBSE_FILE': 'results/wruaresult.db'}


## Define simulation-specific parameters
smin_irrg = ['SSTR_SOIL', 'SFLD_SOIL']
nmmn_cwpr = [100]
rdys_wrua = [1, 2]
abmx_wrua = [1000000000]
flmn_wrua = ['24 * 60 * 60 * 0.01']
abfc_wrua = [1]
fcrs_wrua = [0.0, 0.5, 1.0]
para_simu = list(itertools.product(smin_irrg, nmmn_cwpr, rdys_wrua, abmx_wrua, 
                                   flmn_wrua, abfc_wrua, fcrs_wrua))


## Loop through simulations
for indx, para in enumerate(para_simu):
    
    # Define new dictionary
    para_data = para_temp
    
    # Fill in simulation-specific parameters
    para_data['SIMU']['NMBR_SIMU'] = indx
    para_data['IRRG']['SMIN_IRRG'] = para[0]
    para_data['CWPR']['NMMN_CWPR'] = para[1]
    para_data['WRUA']['RDYS_WRUA'] = para[2]
    para_data['WRUA']['ABMX_WRUA'] = para[3]
    para_data['WRUA']['FLMN_WRUA'] = para[4]
    para_data['WRUA']['ABFC_WRUA'] = para[5]
    para_data['WRUA']['FCRS_WRUA'] = para[6]
    
    # Save as json file
    with open('wruaparam_' + ('%02d' % indx) + '.json', 'w') as para_file:
        json.dump(para_data, para_file, indent = 2)
