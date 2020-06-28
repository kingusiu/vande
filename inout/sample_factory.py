import os
import config.sample_dict as sd


class SamplePathFactory():

    path_dict = {
        'default': sd.base_dir_events,
        'img': sd.base_dir_images,
        'particle': sd.base_dir_events,
        'img-local': (sd.base_dir_images_local, ),
        'particle-local': sd.base_dir_events_local,
    }

    def __init__(self,experiment,mode='default'):
        self.mode = mode
        self.input_dir = self.path_dict[self.mode][0]
        self.result_dir = experiment.result_dir
        if self.mode == 'img-local':
            self.init_img_local()

    def init_img_local(self):
        self.qcd_file_path = os.path.join(self.input_dir,'background_small_img_ptnormal_bin32.h5')
        self.available_samples = ['qcdSide','GtoWW25br','GtoWW35na']
        self.sample_suffix = '_mjj_cut_concat_10K_pt_img.h5'

    @property
    def qcd_path(self):
        return self.qcd_file_path

    def sample_path(self,id):
        if id not in self.available_samples:
            raise 'sample {} not available in mode {}'.format(id,self.mode)
        if id == 'qcdSide':
            return self.qcd_path
        return os.path.join(self.input_dir,sd.file_names[id]+self.sample_suffix)

    def result_path(self,id):
        return os.path.join(self.result_dir,sd.file_names[id]+'_reco.h5')
