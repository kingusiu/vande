import os

base_dir_events = '/eos/user/k/kiwoznia/data/VAE_data/concat_events'
# todo: base dir images

file_names = {
                'qcdSide': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND',
                'qcdSideReco': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND_reco',
                'GtoWW30na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV',
                'GtoWW30br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV',
                'GtoWW35na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV',
                'GtoWW35br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV',
                'GtoWW30naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV_reco',
                'GtoWW30brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_reco',
                'GtoWW35naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_reco',
                'GtoWW35brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_reco',
}

sample_name = {
                'qcdSide': 'QCD side',
                'qcdSideReco': 'QCD side reco',
                'GtoWW30na': r'$G(3.0 TeV)\to WW$ narrow',
                'GtoWW30br': r'$G(3.0 TeV)\to WW$ broad',
                'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
                'GtoWW35br': r'$G(3.5 TeV)\to WW$ broad',
                'GtoWW30naReco': r'$G(3.0 TeV)\to WW$ narrow reco',
                'GtoWW30brReco': r'$G(3.0 TeV)\to WW$ broad reco',
                'GtoWW35naReco': r'$G(3.5 TeV)\to WW$ narrow reco',
                'GtoWW35brReco': r'$G(3.5 TeV)\to WW$ broad reco',

}
