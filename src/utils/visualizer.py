from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize,to_rgb
import matplotlib.patches as mpatches
import numpy as np
import os


# colors = ['black', 'saddlebrown', 'lightgray', 'gray','lavender','blue']
rgb=[]
RGB = [(0 ,0 ,0),(139, 69, 19),(245, 250, 178),(255, 0, 0 ),(85, 255, 255 ),(38 ,40, 210) ]
for i in RGB:
    rgb.append(tuple(j/255 for j in i))




# rgb = [to_rgb(i) for i in colors]

rgber = lambda t: rgb[t]
vfunc = np.vectorize(rgber)


color_labels = ['None', 'Rock', 'Cloud','Shadow','Ice','Water']
cmap = ListedColormap(rgb)

legend_elements = [mpatches.Patch(color=rgb[i], label=color_labels[i]) for i in range(6)]

def blend(class_mask):
    colr_fmask = vfunc(class_mask)
    colr_fmask = np.moveaxis(colr_fmask, 0, 2)

    return colr_fmask

def plot_output(input_img,output,pdf_out_path):

    # def save_results(self,input_img,labels,output):


        fig, a = plt.subplots(1, 2,dpi=200)
        [ai.axis('off') for ai in a]
        a[0].imshow(input_img)
        a[0].set_title('Input')



        a[1].imshow(blend(output))
        a[1].set_title('Predicted')
        fig.legend(handles=legend_elements,loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=2)

        fig.savefig(pdf_out_path, bbox_inches='tight',dpi= 200)
        plt.show()


def get_latex_tables(metrics):
    multicol = r'\multicolumn{2}{c}{'
    closer = r'}'
    CLASS_DICT= {'Clear Land':1,'Cloud':2,'Cloud Shadow':3,'Snow':4,'Water':5}
    for class_name in CLASS_DICT:
        idx1 = '{}_precision'.format(class_name)
        idx2 = '{}_recall'.format(class_name)


        metrics_idx1 = [metrics[i][idx1] for i in range(3)]
        max_val = max(metrics_idx1)
        metrics_idx1 = [r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i) for i in metrics_idx1]

        metrics_idx2 = [metrics[i][idx2] for i in range(3)]
        max_val = max(metrics_idx2)
        metrics_idx2 = [r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i) for i in
                        metrics_idx2]

        print('{} & {}& {}&  {}& {}& {}& {}\\\\'.
              format(class_name, metrics_idx1[0], metrics_idx2[0],
                     metrics_idx1[1], metrics_idx2[1],
                     metrics_idx1[2], metrics_idx2[2],
                     ))
    print(r'\midrule')
    metrics_acc= [metrics[i]['acc'] for i in range(3)]
    max_val = max(metrics_acc)
    metrics_acc = [r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i)
                    for i in metrics_acc]

    print('Total Accuracy & {}{}{}&{}{}{}&{}{}{}\\\\'.
        format(
        multicol, metrics_acc[0], closer,
        multicol, metrics_acc[1], closer,
        multicol, metrics_acc[2], closer,
    ))

    metrics_miou= [metrics[i]['mIOU'] for i in range(3)]
    max_val = max(metrics_miou)
    metrics_miou = [r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i)
                    for i in metrics_miou]

    print('mIOU & {}{}{}&{}{}{}&{}{}{}\\\\'.
        format(
        multicol, metrics_miou[0], closer,
        multicol, metrics_miou[1], closer,
        multicol, metrics_miou[2], closer,
    ))

    print('*' * 20)
    for class_name in CLASS_DICT:
        idx3 = '{}_f1'.format(class_name)
        idx4 = '{}_IOU'.format(class_name)

        metrics_idx3 = [metrics[i][idx3] for i in range(3)]
        max_val = max(metrics_idx3)
        metrics_idx3 = [
            r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i) for i
            in metrics_idx3]

        metrics_idx4 = [metrics[i][idx4] for i in range(3)]
        max_val = max(metrics_idx4)
        metrics_idx4 = [
            r'\bfseries ' + '{:.2}'.format(i) if i == max_val else '{:.2}'.format(i) for i
            in
            metrics_idx4]



        print('{} & {}& {}&  {}& {}& {}& {}\\\\'.
              format(class_name, metrics_idx3[0], metrics_idx4[0],
                     metrics_idx3[1], metrics_idx4[1],
                     metrics_idx3[2], metrics_idx4[2],
                     ))


