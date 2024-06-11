import torch
import numpy as np


def redact(latent_img_batch, coords, orig_x_dim, orig_y_dim=0):
    """
    Takes a latent image batch and returns a redacted version for each image based on provided coordinates.
    It is assumed the latent image is encoded via convolutions and pooling.
    To 'redact', all latent pixels that aren't directly related to the region of interest are set to 0.

    Parameters:
    - latent_img_batch (torch.Tensor - batch_size x num_groups x latent_dim x height x width): 
        Batch of latent images to be modified
    - coords (torch.Tensor - batch_size x num_groups x 4): Tensor containing x1, y1, x2, y2 for each image
    - orig_x_dim, orig_y_dim (int): Dimensions of the original image. Can leave y_dim empty if image is square

    Returns:
    - redacted_latent_img (torch.Tensor - batch_size x num_groups x latent_dim x height x width): 
        Batch of latent images with appropriate regions redacted
    """
    if orig_y_dim == 0:
        orig_y_dim = orig_x_dim

    latent_x_dim, latent_y_dim = latent_img_batch.shape[2:4]
    redacted_latent_img = torch.zeros_like(latent_img_batch)

    for i in range(latent_img_batch.size(0)):
        for j in range(latent_img_batch.size(1)):
            x1, y1, x2, y2 = coords[i][j]

            min_latent_x = int(np.ceil((x1 / orig_x_dim) * latent_x_dim))
            min_latent_y = int(np.ceil((y1 / orig_y_dim) * latent_y_dim))
            max_latent_x = int(np.floor((x2 / orig_x_dim) * latent_x_dim))
            max_latent_y = int(np.floor((y2 / orig_y_dim) * latent_y_dim))

            # Ensuring the region is not out of bounds due to flooring and ceiling operations
            min_latent_x = max(min_latent_x, 0)
            min_latent_y = max(min_latent_y, 0)
            max_latent_x = min(max_latent_x, latent_x_dim - 1)
            max_latent_y = min(max_latent_y, latent_y_dim - 1)

            # Mask the allowed parts over the redacted image for this specific image in the batch
            redacted_latent_img[i, j, :, min_latent_x:max_latent_x+1, min_latent_y:max_latent_y+1] = \
                latent_img_batch[i, j, :, min_latent_x:max_latent_x+1, min_latent_y:max_latent_y+1]

    return redacted_latent_img
