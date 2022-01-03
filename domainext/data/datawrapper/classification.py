from torch.utils.data import Dataset as TorchDataset

from domainext.utils.common import read_image
from ..transforms import to_tensor

__all__ = [
    'BaseWrapper',
    'DomainWrapper'
]

class BaseWrapper(TorchDataset):
    def __init__(self,cfg,data,transform=None,return_img0=True):
        self.cfg = cfg
        self.data = data
        self.transform = transform
        self.return_img0 = return_img0 and cfg.DATALOADER.RETURN_IMG0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        output = self.init_output(item)
        img0 = read_image(item.impath)
        self.transform_img(output,img0)
        if self.return_img0:
            output["img0"] = to_tensor(img0)
        return output

    def init_output(self,item):
        return {
            'label':item.label,
            'impath':item.impath
        }
    
    def transform_img(self,output,img0):
        if self.transform is not None:
            if isinstance(self.transform,(list,tuple)):
                for i,tfm in enumerate(self.transform):
                    img = self._transform_img_one_tfm(tfm,img0)
                    keyname = 'img%s'%(i+1) if i>0 else 'img'
                    output[keyname] = img
        else:
            img = self._transform_img_one_tfm(self.transform,img0)
            output['img'] = img
        
    def _transform_img_one_tfm(self,tfm,img0):
        return tfm(img0)
    
class DomainWrapper(BaseWrapper):
    def __init__(self, cfg, data, transform=None, return_img0=True):
        super().__init__(cfg, data, transform=transform, return_img0=return_img0)
    
    def init_output(self, item):
        return {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath
        }
