import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# the baseline methods
model_names = ['r50', 'r18', 'mob-l', 'mob-s', 'e-b0', 'deit-t']
model_sizes = [25.6, 11.4, 5.4, 2.5, 5.3, 5.]

lincls = [67.5, 51.05, 32.2, 26.78, 42.09, 23.04]
intra_align = [0.9, 0.927, 1.225, 1.303, 1.274, 1.341]
uniformity = [-2.846, -2.743, -3.283, -3.322, -3.505, -3.088]

imp_lincls = [None, 55.72, 47.91, 41.30, 55.89, 38.63]
imp_intra_align = [None, 0.92, 1.29, 1.34, 1.38, 1.43]
imp_uniformity = [None, -2.73, -3.46, -3.47, -3.65, -3.68]

# color maps 
colormap = plt.cm.cool

fig, ax = plt.subplots()

for i, (name, size_m, acc, align, uni) in enumerate(zip(model_names, model_sizes, lincls, intra_align, uniformity)):
    # improved one 
    imp_acc, imp_align, imp_uni = imp_lincls[i], imp_intra_align[i], imp_uniformity[i]
    
    # ax.annotate(f"{name}({acc:.1f}%)", (align, uni))
    if name == 'r50':
        ax.scatter(x=align, y=uni, s=2*size_m**2, color=colormap(acc/100), alpha=0.4)
        ax.annotate(f"{name}({acc:.1f}%)", (align, uni), fontsize=7)
        continue

    # draw the arrow first 
    dx, dy = imp_align - align, imp_uni - uni
    plt.arrow(x=align, y=uni, dx=dx, dy=dy, length_includes_head=True, linestyle=':', alpha=0.2, head_width=0.01)

    ax.scatter(x=align, y=uni, s=size_m**2.5, color=colormap(acc/100), alpha=0.4)
    ax.annotate(f"{name}({acc:.1f}%)", (align, uni), fontsize=5)
    ax.scatter(x=imp_align, y=imp_uni, s=size_m**2.5, color=colormap(imp_acc/100), alpha=1.)
    ax.annotate(f"{imp_acc:.1f}%", (imp_align, imp_uni), fontsize=6)




plt.xlim(0.8, 1.6)
plt.xlabel(r'Intra-Class Alignment$\downarrow$')
plt.ylabel(r'Uniformity$\downarrow$')

# show the colormap of the figure.
fig.colorbar(
    plt.cm.ScalarMappable(cmap=colormap), 
    ax=ax, 
    pad=0.01, 
    aspect=30, 
    format=matplotlib.ticker.PercentFormatter(xmax=1),
    label=r"ImageNet Linear Evaluation Acc(%)$\uparrow$"
    )


fig.savefig('align-uni-space.pdf')