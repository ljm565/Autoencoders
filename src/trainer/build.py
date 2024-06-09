

def get_model(config, device):
    if config.model_type == 'AE':
        from models import AE
        model = AE(config).to(device)
    elif config.model_type == 'CAE':
        from models import CAE
        model = CAE(config).to(device)
    else:
        raise AssertionError(f'Invalid train_type: {config.model_type}')
    return model