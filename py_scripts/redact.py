import numpy as np

def redact(latent_img, x1, y1, x2, y2, orig_x_dim, orig_y_dim=0):
    """
    Takes a latent image and returns a redacted version. It is assumed the latent image is encoded via convolutions and pooling.
    To 'redact', all latent pixels that aren't directly related to the region of interest are set to 0.

    Parameters:
    - latent_img (np array): latent image to be modified
    - x1 & y1 (int): pixel pair denoting the upper left corner of the region of interest
    - x2 & y2 (int): pixel pair denoting the bottom right corner of the region of interest
    - orig_x_dim, orig_y_dim (int): dimensions of the original image. Can leave y_dim empty if image is square

    Returns:
    - redacted_latent_img (np array): A version of the latent image with appropriate regions redacted
    """
    if orig_y_dim == 0:
        orig_y_dim = orig_x_dim
    
    latent_x_dim, latent_y_dim = latent_img.shape[:2]

    min_latent_x = int(np.ceil((x1 / orig_x_dim) * latent_x_dim))
    min_latent_y = int(np.ceil((y1 / orig_y_dim) * latent_y_dim))
    max_latent_x = int(np.floor((x2 / orig_x_dim) * latent_x_dim))
    max_latent_y = int(np.floor((y2 / orig_y_dim) * latent_y_dim))

    # Ensuring the region is not out of bounds due to flooring and ceiling operations
    min_latent_x = max(min_latent_x, 0)
    min_latent_y = max(min_latent_y, 0)
    max_latent_x = min(max_latent_x, latent_x_dim - 1)
    max_latent_y = min(max_latent_y, latent_y_dim - 1)

    # This is the redacted image
    redacted_latent_img = np.zeros_like(latent_img)

    # Mask the allowed parts over the redacted image
    redacted_latent_img[min_latent_x:max_latent_x+1, min_latent_y:max_latent_y+1] = latent_img[min_latent_x:max_latent_x+1, min_latent_y:max_latent_y+1]

    return redacted_latent_img