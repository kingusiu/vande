import os

base_dir_events = '/eos/user/k/kiwoznia/data/VAE_data/concat_events'
# todo: base dir images

file_names = {
                'qcdSide': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND',
                'GtoWW30na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV',
                'GtoWW30br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV',
                'GtoWW35na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV',
                'GtoWW35br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV',
}

sample_name = {
                'qcdSide': 'QCD side',
                'GtoWW30na': r'$G(3.0 TeV)\to WW$ narrow',
                'GtoWW30br': r'$G(3.0 TeV)\to WW$ broad',
                'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
                'GtoWW35br': r'$G(3.5 TeV)\to WW$ broad',

}
