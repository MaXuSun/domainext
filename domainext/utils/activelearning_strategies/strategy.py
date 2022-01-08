import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainext.data.loader import fast_build_test_loader

class Strategy:
    def __init__(self, wrapper_labeled, wrapper_unlabeled, model, num_classes,embedding_dim,**kwargs):
        """ The query strategies for selecting samples for Active Learning

        Args:
            wrapper_labeled ([Pytorch Dataset]): The labeled pytorch dataset(called wrapper in this projection.)
            wrapper_unlabeled ([Pytorch Dataset]): The unlabeled pytorch dataset.
            model ([Module or dict[str:Module]]): The model for strategies, which can be a nn.Moudle or a dict composed of nn.Module. 
            num_classes ([int]): The number of categories.
            embedding_dim ([int]): The embedding dimensionality of model.
            args[dict]: Stroe some optinal parameters.
                inference_func[function]: A customized model inference function, which default value is 'return model(input)'
                batch_size[int]: Batch size, which default value is 1.
                device[torch.Device]: 
                loss_func[function]: Loss function, which default value is CrossEntropy loss function. 
        """

        self.wrapper_labeled = wrapper_labeled
        self.wrapper_unlabeled = wrapper_unlabeled
        self.model = model
        self.target_classes = num_classes
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs

        if 'inference_func' not in kwargs and kwargs['inference_func'] is not None:
            self.inference_func = self.model_inference
        else:
            self.inference_func = kwargs['inference_func']

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 1

        if 'device' not in kwargs:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = kwargs['device']

        if 'loss_func' not in kwargs:
            self.loss = F.cross_entropy
        else:
            self.loss = kwargs['loss_func']

    def select(self, budget):
        raise NotImplemented
        
    def model_inference(self, model, input, return_feature=False, freeze=False):
        return model(input, return_feature=return_feature, freeze=freeze)

    def set_model_mode(self, mode='eval'):
        if isinstance(self.model, dir):
            for key in self.model:
                if mode == 'eval':
                    self.model[key].eval()
                else:
                    self.model[key].train()
        else:
            if mode == 'eval':
                self.model.eval()
            else:
                self.model.train()

    def update_data(self, wrapper_labeled, wrapper_unlabeled):
        self.wrapper_labeled = wrapper_labeled
        self.wrapper_unlabeled = wrapper_unlabeled

    def update_queries(self, wrapper_query):
        self.wrapper_query = wrapper_query

    def update_model(self, clf):
        self.model = clf

    def predict(self, to_predict_wrapper):

        # Ensure model is on right device and is in eval. mode
        self.set_model_mode('eval')

        # Create a tensor to hold class predictions
        P = torch.zeros(len(to_predict_wrapper)).long().to(self.device)

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)

        evaluated_instances = 0

        with torch.no_grad():
            for _, elements_to_predict in enumerate(to_predict_dataloader):

                # Predict the most likely class
                elements_to_predict = elements_to_predict['img'].to(self.device)
                out = self.inference_func(self.model, elements_to_predict)
                pred = out.max(1)[1]

                # Insert the calculated batch of predictions into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                P[start_slice:end_slice] = pred
                evaluated_instances = end_slice

        return P

    def predict_prob(self, to_predict_wrapper):

        # Ensure model is on right device and is in eval. mode
        self.set_model_mode('eval')

        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_wrapper), self.target_classes]).to(self.device)

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)

        evaluated_instances = 0

        with torch.no_grad():
            for _, elements_to_predict in enumerate(to_predict_dataloader):

                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict['img'].to(self.device)
                out = self.inference_func(self.model, elements_to_predict)
                pred = F.softmax(out, dim=1)

                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                probs[start_slice:end_slice] = pred
                evaluated_instances = end_slice

        return probs

    def predict_prob_dropout(self, to_predict_wrapper, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.set_model_mode('train')

        # Create a tensor to hold probabilities
        probs = torch.zeros([len(to_predict_wrapper), self.target_classes]).to(self.device)

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):

                evaluated_instances = 0
                for _, elements_to_predict in enumerate(to_predict_dataloader):

                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict['img'].to(self.device)
                    out = self.inference_func(self.model, elements_to_predict)
                    pred = F.softmax(out, dim=1)

                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[start_slice:end_slice] += pred
                    evaluated_instances = end_slice

        # Divide through by n_drop to get average prob.
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, to_predict_wrapper, n_drop):

        # Ensure model is on right device and is in TRAIN mode.
        # Train mode is needed to activate randomness in dropout modules.
        self.set_model_mode('train')

        # Create a tensor to hold probabilities
        probs = torch.zeros([n_drop, len(to_predict_wrapper), self.target_classes]).to(self.device)

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)

        with torch.no_grad():
            # Repeat n_drop number of times to obtain n_drop dropout samples per data instance
            for i in range(n_drop):

                evaluated_instances = 0
                for _, elements_to_predict in enumerate(to_predict_dataloader):

                    # Calculate softmax (probabilities) of predictions
                    elements_to_predict = elements_to_predict['img'].to(self.device)
                    out = self.inference_func(self.model, elements_to_predict)
                    pred = F.softmax(out, dim=1)

                    # Accumulate the calculated batch of probabilities into the tensor to return
                    start_slice = evaluated_instances
                    end_slice = start_slice + elements_to_predict.shape[0]
                    probs[i][start_slice:end_slice] = pred
                    evaluated_instances = end_slice

        return probs

    def get_embedding(self, to_predict_wrapper):

        # Ensure model is on right device and is in eval. mode
        self.set_model_mode('eval')

        # Create a tensor to hold embeddings
        embedding = torch.zeros([len(to_predict_wrapper), self.embedding_dim]).to(self.device)

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)

        evaluated_instances = 0

        with torch.no_grad():

            for _, elements_to_predict in enumerate(to_predict_dataloader):

                # Calculate softmax (probabilities) of predictions
                elements_to_predict = elements_to_predict['img'].to(self.device)
                out, l1 = self.inference_func(self.model, elements_to_predict, return_feature=True)

                # Insert the calculated batch of probabilities into the tensor to return
                start_slice = evaluated_instances
                end_slice = start_slice + elements_to_predict.shape[0]
                embedding[start_slice:end_slice] = l1
                evaluated_instances = end_slice

        return embedding

    # gradient embedding (assumes cross-entropy loss)
    # calculating hypothesised labels within
    def get_grad_embedding(self, to_predict_wrapper, predict_labels, grad_embedding_type="bias_linear"):
        self.set_model_mode('train')

        # Create the tensor to return depending on the grad_embedding_type, which can have bias only,
        # linear only, or bias and linear
        if grad_embedding_type == "bias":
            grad_embedding = torch.zeros([len(to_predict_wrapper), self.target_classes]).to(self.device)
        elif grad_embedding_type == "linear":
            grad_embedding = torch.zeros([len(to_predict_wrapper), self.embedding_dim * self.target_classes]).to(self.device)
        elif grad_embedding_type == "bias_linear":
            grad_embedding = torch.zeros([len(to_predict_wrapper), (self.embedding_dim + 1) * self.target_classes]).to(self.device)
        else:
            raise ValueError("Grad embedding type not supported: Pick one of 'bias', 'linear', or 'bias_linear'")

        # Create a dataloader object to load the dataset
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper, bs=self.kwargs['batch_size'])

        evaluated_instances = 0

        # If labels need to be predicted, then do so. Calculate output as normal.
        for _, unlabeled_data_batch in enumerate(to_predict_dataloader):
            start_slice = evaluated_instances
            end_slice = start_slice + unlabeled_data_batch.shape[0]

            inputs = unlabeled_data_batch['img'].to(self.device, non_blocking=True)
            targets = unlabeled_data_batch['label'].to(self.device, non_blocking=True)

            out, l1 = self.inference_func(self.model, inputs, return_feature=True, freeze=True)
            if predict_labels:
                targets = out.max(1)[1]

            # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
            loss = self.loss(out, targets, reduction="sum")
            l0_grads = torch.autograd.grad(loss, out)[0]

            # Calculate the linear layer gradients as well if needed
            if grad_embedding_type != "bias":
                l0_expand = torch.repeat_interleave(l0_grads, embedding_dim, dim=1)
                l1_grads = l0_expand * l1.repeat(1, self.target_classes)

            # Populate embedding tensor according to the supplied argument.
            if grad_embedding_type == "bias":
                grad_embedding[start_slice:end_slice] = l0_grads
            elif grad_embedding_type == "linear":
                grad_embedding[start_slice:end_slice] = l1_grads
            else:
                grad_embedding[start_slice:end_slice] = torch.cat([l0_grads, l1_grads], dim=1)

            evaluated_instances = end_slice

            # Empty the cache as the gradient embeddings could be very large
            torch.cuda.empty_cache()

        # Return final gradient embedding
        return grad_embedding

    def feature_extraction(self, inp, layer_name):
        feature = {}
        model = self.model

        def get_features(name):
            def hook(model, inp, output):
                feature[name] = output.detach()
            return hook

        if isinstance(self.model, dir):
            for key in self.model:
                temp_model = self.model[key]
                for name, layer in temp_model._modules.items():
                    if name == layer_name:
                        layer.register_forward_hook(get_features(layer_name))
        else:
            for name, layer in self.model._modules.items():
                if name == layer_name:
                    layer.register_forward_hook(get_features(layer_name))
        output = self.inference_func(self.model, inp)
        return torch.squeeze(feature[layer_name])

    def get_feature_embedding(self, to_predict_wrapper, layer_name='avgpool'):
        to_predict_dataloader = fast_build_test_loader(to_predict_wrapper)
        features = []
        for _, batch in enumerate(to_predict_dataloader):
            inputs = batch['img'].to(self.device)
            batch_features = self.feature_extraction(inputs, layer_name)
            features.append(batch_features)
        return torch.vstack(features)

