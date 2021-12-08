import torch
import torch.nn.functional as F
import model.utils as mutils
from sampling import get_predictor, get_corrector
from model.feature_extractor import FeatureExtractor
import pdb
from tqdm import tqdm

def get_image_captioning_grad_fn(sde, img_cap_model, max_scale=None):

    feature_extractor = FeatureExtractor()
    
    def image_captioning_grad_fn(x, t, text):
        with torch.enable_grad():
            x = x.clone().detach()
            x.requires_grad = True
            
            #Scale input image x to [0, 1]
            max = torch.ones(x.shape[0], device='cuda:0')
            min = torch.ones(x.shape[0], device='cuda:0')
            for N in range(x.shape[0]):
                max[N] = torch.max(x[N, :, :, :])
                min[N] = torch.min(x[N, :, :, :])
            x = x - min[:, None, None, None] * torch.ones_like(x, device='cuda:0')
            x = torch.div(x, (max - min)[:, None, None, None])
                        
            # _, std = sde.marginal_prob(x, t)
            img_feature = feature_extractor(x)
            
            # pred is the predicted captions for image
            seq, seq_logprobs = img_cap_model(img_feature.mean(0)[None], img_feature[None], 
                                 mode='sample',
                                 opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':1})
            # seq = seq.data
            # Logits to probabilities
            seq_logprobs = F.softmax(seq_logprobs, dim=2)
            seq_len = text.size(1)
            text_indices = torch.unsqueeze(text, -1)
            prob = torch.gather(seq_logprobs[:, :seq_len, :], dim=2, index=text_indices)
            prob = torch.sum(prob.squeeze(-1), dim=-1)
            # prob = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq>0).to(seq_logprobs).sum(1)+1)
            
            # prob = F.softmax(seq_logprobs, dim=2)
            #print('prob: ', prob.shape)
            #print("is nan: ", torch.isnan(prob).any())
            #prob, _ = torch.max(prob, dim=2)
            # prob = torch.sum(prob)
            #print('prob_sum: ', prob.shape)
            #print('x: ', x.shape)
            
            #Reduce probabilities to only the ones matching the original label at one pixel
            # prob = torch.mul(prob, sem_mask)
            #Removing channel dimension by discarding all 0 elements from above operation
            # prob, _ = torch.max(prob, dim=1)
            
            # print(x.is_leaf)
            grad = torch.autograd.grad(prob, x, torch.ones_like(prob), 
                                       create_graph=True, allow_unused=True)
            # grad = torch.autograd.grad(prob, x, torch.ones_like(prob), allow_unused=True)
        
            # print('grad: ', grad[0])
        return grad[0]
    
    return image_captioning_grad_fn

def get_text2image_sampler(config, sde, score_model, img_cap_model, text, n_steps=1,
                           probability_flow=False, denoise=True, eps=1e-3, device='cuda:0', scale=None):
    """
    Gets the sampler for semantic synthesis
    :param sde: The 'sde_lib.SDE' of the model
    :param score_model: The score model
    :param img_cap_model: The image captioning model trained for the dataset
    :param text: The text to generate a sample from (in format xxx)
    :param n_steps: The corrector steps per corrector update
    :param probability_flow: If predictor should use probability flow
    :param denoise: If true denoises the sample before returning it
    :param eps: A Number for numerical stability
    :param device: PyTorch device
    :return: The sampling function
    """
    #Create gradient functions
    score_fn = mutils.get_score_fn(sde, score_model, train=False)
    # need to change get_semantic_segmentation_grad_fn to get_image_captioning_grad_fn
    image_captioning_grad_fn = get_image_captioning_grad_fn(sde, img_cap_model, 
                                                config.sampling.img_cap_scale if scale is None else scale)
    
    # Get predictor and corrector
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())

    def total_grad_fn(x, t):
        #print('score: ', score_fn(x, t))
        # print('image_captioning: ', image_captioning_grad_fn(x, t))
        k = 10000
        return score_fn(x, t) + k * image_captioning_grad_fn(x, t, text)
        # return score_fn(x, t)

    # Create predictor & corrector update functions
    def predictor_update_fn(x, t):
        predictor_inst = predictor(sde, total_grad_fn, probability_flow)
        return predictor_inst.update_fn(x, t)
    
    def corrector_update_fn(x, t):
        corrector_inst = corrector(sde, total_grad_fn, config.sampling.snr, n_steps)
        return corrector_inst.update_fn(x, t)

    # Define sampler process
    shape = (config.sampling.n_samples_per_text, 3, 32, 32)
    def text2image_sampler():
        """
        Generate text2image samples with Predictor-Corrector (PC) samplers.
        :return: Samples
        """
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            
            timesteps = torch.linspace(1, eps, sde.N, device=device)
            for i in tqdm(range(sde.N)):
                vec_t = torch.ones(config.sampling.n_samples_per_text, device=device) * timesteps[i]
                x, x_mean = corrector_update_fn(x, vec_t)
                x = x.float()
                x_mean = x.float()
                x, x_mean = predictor_update_fn(x, vec_t)
                x = x.float()
                x_mean = x.float()
            return x_mean if denoise else x

    return text2image_sampler