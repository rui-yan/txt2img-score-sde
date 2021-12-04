from configs.default_coco_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vesde'

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # text sampling
    sampling.sample_data_dir = '/data/yan/score-based/opt/cocoapi/images/test2014/'
    sampling.img_seg_model_dir = '/home/yan/score-sde/text2image/image_captioning/model-best.pth'
    sampling.n_samples_per_text = 1

    # data
    data = config.data

    # model
    model = config.model
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.0
    model.embedding_type = 'positional'
    model.conv_size = 3
    
    # model.scale_by_sigma = True
    # model.ema_rate = 0.999
    # model.normalization = 'GroupNorm'
    # model.nonlinearity = 'swish'
    # model.nf = 128
    # # model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    # # model.n_res_blocks = 2
    # model.ch_mult = (1, 2, 2, 2)
    # model.num_res_blocks = 4
    # model.attn_resolutions = (16,)
    # model.resamp_with_conv = True
    # model.conditional = True
    # model.fir = True
    # model.fir_kernel = [1, 3, 3, 1]
    # model.skip_rescale = True
    # model.resblock_type = 'biggan'
    # model.progressive = 'none'
    # model.progressive_input = 'residual'
    # # model.progressive = 'output_skip'
    # # model.progressive_input = 'input_skip'
    # model.progressive_combine = 'sum'
    # model.attention_type = 'ddpm'
    # model.init_scale = 0.
    # model.fourier_scale = 16
    # model.conv_size = 3
    # model.sampling_eps = 1e-5

    return config