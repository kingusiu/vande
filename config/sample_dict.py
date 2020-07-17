import os

base_dir_events = '/eos/user/k/kiwoznia/data/VAE_data/concat_events'
base_dir_images = '/eos/user/k/kiwoznia/data/VAE_data/march_2020_data/input/images'
base_dir_results = '/eos/home-k/kiwoznia/dev/autoencoder_for_anomaly/convolutional_VAE/results'

base_dir_events_local = 'data/events'
base_dir_images_local = 'data/images'
base_dir_results_local = 'results'


file_names = {
                'qcdSide': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND',
                'qcdSideReco': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND_reco',
                'qcdSig': 'qcd_sqrtshatTeV_13TeV_PU40',
                'qcdSigReco': 'qcd_sqrtshatTeV_13TeV_PU40_reco',
                'GtoWW15na': 'RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV',
                'GtoWW15br': 'RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV',
                'GtoWW25na': 'RSGraviton_WW_NARROW_13TeV_PU40_2.5TeV',
                'GtoWW25br': 'RSGraviton_WW_BROAD_13TeV_PU40_2.5TeV',
                'GtoWW30na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV',
                'GtoWW30br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV',
                'GtoWW35na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV',
                'GtoWW35br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV',
                'GtoWW45na': 'RSGraviton_WW_NARROW_13TeV_PU40_4.5TeV',
                'GtoWW45br': 'RSGraviton_WW_BROAD_13TeV_PU40_4.5TeV',
                'GtoWW15naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_reco',
                'GtoWW15brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV_reco',
                'GtoWW25naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_2.5TeV_reco',
                'GtoWW25brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_2.5TeV_reco',
                'GtoWW30naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_3.0TeV_reco',
                'GtoWW30brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_reco',
                'GtoWW35naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_reco',
                'GtoWW35brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_reco',
                'GtoWW45naReco': 'RSGraviton_WW_NARROW_13TeV_PU40_4.5TeV_reco',
                'GtoWW45brReco': 'RSGraviton_WW_BROAD_13TeV_PU40_4.5TeV_reco',
}

sample_name = {
                'qcdSide': 'QCD side',
                'qcdSideReco': 'QCD side reco',
                'qcdSig': 'QCD signalregion',
                'qcdSigReco': 'QCD signalregion reco',
                'GtoWW15na': r'$G(1.5 TeV)\to WW$ narrow',
                'GtoWW15br': r'$G(1.5 TeV)\to WW$ broad',
                'GtoWW25na': r'$G(2.5 TeV)\to WW$ narrow',
                'GtoWW25br': r'$G(2.5 TeV)\to WW$ broad',
                'GtoWW30na': r'$G(3.0 TeV)\to WW$ narrow',
                'GtoWW30br': r'$G(3.0 TeV)\to WW$ broad',
                'GtoWW35na': r'$G(3.5 TeV)\to WW$ narrow',
                'GtoWW35br': r'$G(3.5 TeV)\to WW$ broad',
                'GtoWW45na': r'$G(4.5 TeV)\to WW$ narrow',
                'GtoWW45br': r'$G(4.5 TeV)\to WW$ broad',
                'GtoWW15naReco': r'$G(1.5 TeV)\to WW$ narrow reco',
                'GtoWW15brReco': r'$G(1.5 TeV)\to WW$ broad reco',
                'GtoWW25naReco': r'$G(2.5 TeV)\to WW$ narrow reco',
                'GtoWW25brReco': r'$G(2.5 TeV)\to WW$ broad reco',
                'GtoWW30naReco': r'$G(3.0 TeV)\to WW$ narrow reco',
                'GtoWW30brReco': r'$G(3.0 TeV)\to WW$ broad reco',
                'GtoWW35naReco': r'$G(3.5 TeV)\to WW$ narrow reco',
                'GtoWW35brReco': r'$G(3.5 TeV)\to WW$ broad reco',
                'GtoWW45naReco': r'$G(4.5 TeV)\to WW$ narrow reco',
                'GtoWW45brReco': r'$G(4.5 TeV)\to WW$ broad reco',
}
