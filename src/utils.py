import torch
import huffman

def calc_avg_bits(
    num_codebooks: int = 8,
    out_group_size: int = 1,
    in_group_size: int = 32,
    nbits_per_codebook=8,
    in_features: int = 8192,
    out_features: int = 8192,
):
    '''
    Calculates average bits for parameters in one layer.
    :param num_codebooks: codebook quantity
    :param out_group_size: size of groupes in out dimension
    :param in_group_size: size of groupes in in dimension
    :param nbits_per_codebook: number of bits for codebook
    :param in_features: shape of input dimension of layer
    :param out_features: shape of output dimension of layer
    
    '''
    codebook_size = 2**nbits_per_codebook #codebook size 
    codebooks_store = num_codebooks * codebook_size * out_group_size * in_group_size * 16  # bits
    matrix_store = (
        out_features * in_features // (out_group_size * in_group_size) * num_codebooks * nbits_per_codebook
    )  # bits
    # avg bits per parameter
    return (matrix_store + codebooks_store) / (in_features * out_features)

def get_mean_nbits_by_codebook(codes: torch.IntTensor, huffman_group_size: int = 2):

    '''
    Calculates average code length in codebooks.
    :param codes: codebook codes
    :param huffman_group_size: huffman compresssion dimension count
    '''
    _, codebook_size, num_codebooks = codes.shape
    flat_codes_by_codebook = codes.permute(2, 0, 1).flatten(1, 2)
    code_counts = torch.zeros(num_codebooks, codebook_size, device=flat_codes_by_codebook.device,
                                   dtype=flat_codes_by_codebook.dtype).scatter_add(
        -1, flat_codes_by_codebook, torch.ones_like(flat_codes_by_codebook)
    )  # shape: [current beam_size, num_codebooks, codebook_size], initial beam_size = 1
    code_probs = code_counts / code_counts.sum(dim=-1, keepdim=True).float()
    code_probs = code_probs.cpu().numpy()
    assert num_codebooks % huffman_group_size == 0

    mean_code_lengths = []
    for group_index in range(num_codebooks // huffman_group_size):
        group_code_probs = {(): 1}

        for codebook_index in range(group_index * huffman_group_size, (group_index + 1) * huffman_group_size):
            new_group_code_probs = {}
            for group, group_prob in group_code_probs.items():
                for code, code_prob in tuple(enumerate(code_probs[codebook_index])):
                    new_group_code_probs[group + (code,)] = group_prob * code_prob
            group_code_probs = new_group_code_probs


        huffman_codebook_i = huffman.codebook(list(group_code_probs.items()))
        codebook_mean_code_length_i = sum(
            len(huffman_codebook_i[code]) * prob for code, prob in group_code_probs.items()
        )
        mean_code_lengths.append(codebook_mean_code_length_i)
    return mean_code_lengths