import matplotlib.pyplot as plt
import numpy as np

def visualise_L(L_orig, L_opt):
#     L_orig = -(L_orig - np.diag(np.diag(L_orig)))
#     L_opt = -(L_opt - np.diag(np.diag(L_opt)))
    
    ax1 = plt.subplot(121)
    ax1.imshow(L_orig)
    ax1.title.set_text(r"True $L$")
    ax2 = plt.subplot(122)
    ax2.imshow(L_opt)
    ax2.title.set_text(r"Predicted $L$")
    plt.show()