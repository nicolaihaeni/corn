
def get_renderer(renderer='synsin', options=None):
    ''' Function to create different renderers based on specificaiton
    '''
    if renderer == 'synsin':
        from models.renderer.synsin_renderer import SynSinRenderer
        return SynSinRenderer(n_filters=options.n_filters, radius=2, img_size=options.img_size,
                              points_per_pixel=16, gamma=1.0)
    else:
        raise NotImplementedError()
