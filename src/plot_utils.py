import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_aux(plt, title, output_folder='', output_prefix='',
             legend_pos=None, fontsize=None, save_also_without_title=False, local_show=True, 
             suptitle_instead_of_title=False, bbox_extra_artistslist=None, extra_args={}):
    '''
    a wrapper for plt.show() and plt.savefig()
    @ plt: plt object
    @ title: title of the plot, which also uses as the name of the file if output_folder is not empty string
    @ output_folder: if not empty string, save the plot to this folder
    @ fontsize: fontsize of the legend
    @ output_prefix: prefix of the file name if save the plot
    @ save_also_without_title: save the plot with and without title
    @ legend_pos: position of the legend
    @ local_show: show the plot locally
    '''
    if fontsize is not None:
        plt.legend(fontsize=f"{fontsize}", loc=legend_pos)
        try:
            # get current xlabel name
            plt.xlabel(plt.gca().get_xlabel(), fontsize=fontsize)
            plt.ylabel(plt.gca().get_ylabel(), fontsize=fontsize)
        except:
            pass
    
    bbox_extra_artistslist = None
    if bbox_extra_artistslist or 'bbox_extra_artistslist' in extra_args:
        bbox_extra_artistslist = extra_args['bbox_extra_artistslist']

    if suptitle_instead_of_title or 'suptitle_instead_of_title' in extra_args:
        plt.suptitle(title)
    else:
        plt.title(title)  
    if type(output_folder) == str and output_folder != '':
        save_to = os.path.join(output_folder, repr(output_prefix + title.replace(' ', '_').replace('/n', '_').replace('/', '_')).strip("'"))
        if not save_to.endswith('.png'):
            save_to = save_to + '.png'
        try:
            plt.savefig(save_to, bbox_extra_artists=bbox_extra_artistslist, bbox_inches='tight')
        except Exception as e:
            print(f'failed to save plot {save_to} due {e}')

        if save_also_without_title:
            # plt.title('')
            if suptitle_instead_of_title or 'suptitle_instead_of_title' in extra_args:
                plt.suptitle('')
            else:
                plt.title('')  

            # plt.tight_layout()
            try:
                plt.savefig(save_to.replace('.png', '_no_title.png'), bbox_extra_artists=bbox_extra_artistslist, bbox_inches='tight')
            except Exception as e:
                print(f'failed to save plot {save_to} due {e}')


    if local_show:
        if suptitle_instead_of_title or 'suptitle_instead_of_title' in extra_args:
            plt.suptitle(title)
        else:
            plt.title(title)   
        plt.show()
    plt.clf()


def plot_aux_wrapper(output_folder='', output_prefix='',
             legend_pos=None, fontsize=None, save_also_without_title=False, 
             suptitle_instead_of_title=False,
             local_show=True, extra_args={}):
    if not os.path.exists(output_folder) or not os.path.isdir(output_folder):
        raise ValueError(f'output_folder {output_folder} is not a valid folder')
    def wrapper(plt, title, output_folder=output_folder, output_prefix=output_prefix,
             legend_pos=legend_pos, fontsize=fontsize, save_also_without_title=save_also_without_title, 
             local_show=local_show, suptitle_instead_of_title=suptitle_instead_of_title, extra_args=extra_args):
        return plot_aux(plt, title, output_folder=output_folder, output_prefix=output_prefix,
                legend_pos=legend_pos, fontsize=fontsize, save_also_without_title=save_also_without_title, 
                local_show=local_show, suptitle_instead_of_title=suptitle_instead_of_title, 
                extra_args=extra_args)
    return wrapper
