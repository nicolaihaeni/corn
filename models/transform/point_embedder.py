import torch
import torch.nn as nn


'''
Helper function to create point embedding function
'''
def get_point_embedder(n_freq=10, i=0):
    if i==-1:
        return nn.Identidy(), 3

    embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': n_freq-1,
            'n_freq': n_freq,
            'log_sampling': False,
            'periodic_fns': [torch.sin, torch.cos]
            }

    embedder_obj = PointEmbedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class PointEmbedder:
    '''
    Higher dimensional point embedding inspired by: NeRF: Representing Scenes as Neural Radiance     Fields for View Synthesis, https://arxiv.org/abs/2003.08934
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        dims = self.kwargs['input_dims']
        max_freq = self.kwargs['max_freq_log2']
        n_freq = self.kwargs['n_freq']
        out_dim = 0

        embed_fns = []
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += dims

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, n_freq)
        else:
            freq_bands = torch.linspace(2**0., 2.**max_freq, n_freq)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
