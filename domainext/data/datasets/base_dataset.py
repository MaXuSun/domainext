from domainext.utils.common import check_isfile
import os
import os.path as osp
import tarfile
import zipfile
import gdown

__all__ = [
    'LableDatum',
    'FullDatum',
    'ClassDatasetBase',
    'SegDatasetBase'
]
#############
### Datum ###
#############

class BaseDatum:
    """Data instance which defines the basic attributes.
    Args:
        impath (str): image path.
    """
    def __init__(self, impath=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)
        self._impath = impath

    @property
    def impath(self):
        return self._impath

    def __eq__(self,other):
        return self.__dict__ == other.__dict__
    
    def __str__(self):
        return self._impath

class LabelDatum(BaseDatum):
    def __init__(self, impath="",label=None):
        super().__init__(impath=impath)
        if isinstance(label,str):
            assert check_isfile(label)
        self._label = label
    @property
    def label(self):
        return self._label
    
    def __str__(self):
        return " ".join(self._impath,str(self._label))

class FullDatum(BaseDatum):
    def __init__(self, impath="", label=None, domain=0,classname=""):
        super().__init__(impath=impath)
        self._label = label
        self._domain = domain
        self._classname = classname
    @property
    def classname(self):
        return self._classname
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def label(self):
        return self._label

    def __str__(self):
        return " ".join(self._impath,str(self._label),str(self._domain),self._classname)

###############
### Dataset ###
###############

class DatasetBase:
    """
    A unified base dataset class
    """
    dataset_dir = ""
    domains = []

    def __init__(self,train_x=None,train_u=None,val=None,test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print("File extracted to {}".format(osp.dirname(dst)))

    def add_item(self,item,name='train_x',record=False):
        if record:
            print('Add item: ',item)
        getattr(self,name).append(item)
    
    def add_items(self,items,name='train_x',record=True):
        if record:
            for item in items:
                print('Add item: ',item)
        getattr(self,name).extend(items)
    
    def remove_item(self,item,name='train_x',record=True):
        if record:
            print('Remove item: ',item)
        getattr(self,name).remove(item)
    
    def remove_items(self,items,name='train_x',record=False):
        for item in items:
            self._remove_item(item,name,record)
    
    def set_data(self,newdata,name):
        setattr(self,name,newdata)

    def _read_json_xu(self):
        pass
    def _write_json_xu(self):
        pass

class ClassDatasetBase(DatasetBase):
    """A dataset class for classification.
    """
    num_classes = 0
    
    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
        self._num_classes = self.num_classes
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames
    
class SegDatasetBase(DatasetBase):
    """A dataset class for segmentation.
    """
    pass
