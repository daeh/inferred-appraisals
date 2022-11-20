

def plotDM(ax, sns, dm, mask=None, vmin=None, vmax=None, vcenter=None, cmap='YlGnBu', label_rename_dict=None, show_labels=True):
    # https://stackoverflow.com/questions/47916205/seaborn-heatmap-move-colorbar-on-top-of-the-plot
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.colorbar.html
    # https://joseph-long.com/writing/colorbars/

    import numpy as np

    if vmin is None:
        vmin = np.floor(dm.min().min())
    if vmax is None:
        vmax = np.ceil(dm.max().max())
    if vcenter is None:
        vcenter = (vmax - vmin) / 2
    if mask is None:
        mask = np.full_like(dm, True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(dm, ax=ax, mask=mask, vmin=vmin, vmax=vmax, center=vcenter, cmap=cmap,
                     square=True, linewidths=.5, linecolor='white',
                     cbar=False,
                     #  cbar_kws={"orientation": "horizontal", "ticks":[vmin,vcenter,vmax]}, #"shrink": 1
                     )

    ax.set_axisbelow(False)

    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off

    if label_rename_dict is None:
        xlabels_val = dm.columns.to_list()
        ylabels_val = dm.index.to_list()
    else:
        xlabels_val = [label_rename_dict[label] for label in dm.columns.to_list()]
        ylabels_val = [label_rename_dict[label] for label in dm.index.to_list()]

    if show_labels:

        ax.set_xticks(range(dm.shape[1]))
        ax.set_yticks(range(dm.shape[0]))
        ax.set_xticklabels(xlabels_val)
        ax.set_yticklabels(ylabels_val)

        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()

        for ilabel, xlabel in enumerate(xlabels):
            ax.text(ilabel + 0.25, ilabel, xlabel.get_text(), horizontalalignment='left', verticalalignment='bottom', rotation=45, fontsize=9)  # weight='semibold'# transform=ax.transAxes
            ax.text(-0.3, ilabel + 0.5, xlabel.get_text(), horizontalalignment='right', verticalalignment='center', rotation=0, fontsize=9)  # weight='semibold'  #

    ax.set_xticklabels([''] * len(xlabels_val))
    ax.set_yticklabels([''] * len(ylabels_val))

    return ax


def plotDM_handler(dm, label_rename_dict=None, plotParam=None, outpath=None, show_labels=True, show_colorbar=True, invert_mask=False):
    import numpy as np
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    assert len(np.unique(dm.shape)) == 1

    plt = plotParam['plt']
    sns = plotParam['sns']

    plt.close('all')

    cmap1 = 'YlGnBu'
    cmap2 = sns.diverging_palette(220, 10, as_cmap=True)
    cmap3 = sns.diverging_palette(250, 15, s=75, l=40, center="dark", as_cmap=True)
    cmap4 = sns.diverging_palette(200, 300, s=75, l=40, as_cmap=True)

    mask_allemotions_tril1 = np.zeros((dm.shape[0], dm.shape[0]), dtype=bool)
    mask_allemotions_tril1[np.triu_indices_from(mask_allemotions_tril1, k=1)] = True
    if invert_mask:
        mask_allemotions_tril1 = ~mask_allemotions_tril1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = plotDM(ax, sns, dm, mask=mask_allemotions_tril1, vmin=-1, vmax=1, vcenter=0, cmap=cmap4, label_rename_dict=label_rename_dict, show_labels=show_labels)

    if show_colorbar:
        ### colorbar ###
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('bottom', size='4%', pad='2%')
        fig.colorbar(ax.get_children()[0], ax=ax, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
        cax.xaxis.set_ticks_position('bottom')

    figsout = list()
    figsout.append((outpath, fig, True))
    plt.close(fig)

    return figsout


def UNUSED_singlemodel_singledataset_plots(model_results, dsname, savepath, emotions, outcomes, plotParam):
    import numpy as np

    modelfigsout = savepath / dsname

    figsout = list()

    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']
    plt = plotParam['plt']
    sns = plotParam['sns']

    # if i_ds == 0:
    #     figout, ax = plt.subplots(figsize=(8, 48))
    #     plotFun3(model_results['dissimilarity']['dm']['empirical'], emotions, ax, sns)
    #     # ax.set_title(''.format())
    #     # ax.set_xticklabels()
    #     figsout.append( (savepath / 'DM-linear-empirical-contast.pdf'.format(), figout) )
    #     plt.close('all')

    ######################################################################

    n_emotions = len(emotions)
    mask_allemotions_tril1 = np.zeros((n_emotions, n_emotions), dtype=bool)
    mask_allemotions_tril1[np.triu_indices_from(mask_allemotions_tril1, k=1)] = True
    mask = mask_allemotions_tril1

    dmemp_mean = model_results['dissimilarity']['dm']['empirical']['mean_byoutcome']

    #####################################

    # figout,axs = plt.subplots(2, 2, figsize=(20,20))
    for i_outcome, outcome in enumerate(outcomes):
        figout, ax = plt.subplots(figsize=(11, 9))
        # ax = axs[[(0,0),(0,1),(1,0),(1,1)][i_outcome]]
        plotDM_handler(dmemp_mean[outcome], ax, sns)
        ax.text(ax.get_xlim()[1] - 0.5, 0.5, outcome, horizontalalignment='right', verticalalignment='top', rotation=0, fontsize=42, weight='semibold')  # transform=ax.transAxes

        figsout.append((savepath / 'empirical' / 'DM-empirical_withinOutcome_MEANacrossPot-{}.pdf'.format(outcome), figout))
        plt.close(figout)

    return figsout


def new_emo_plot(ppldataemodf, emoevdf=None, scale_factor=1.0, bandwidth=0.05, emotions_abbriv=None, fig_outpath=None, plotParam=None):

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    """
    e.g.
    ppldataemodf = ppldata['empiricalEmotionJudgments']
    """
    # %%

    def gradient_image(ax, X, extent, direction=0.0, cmap_range=(0, 1), **kwargs):
        """
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* keyword argument.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.
        """

        im = ax.imshow(np.flipud(X), extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, **kwargs)
        return im

    plt = plotParam['plt']

    outcomes = plotParam['outcomes']

    emotions = ppldataemodf['CC'].loc[:, 'emotionIntensities'].columns.tolist()

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(9.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(2.0), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    legend_str = dict(CC="CC (0.5, 0.5)", CD="CD (0, 1)", DC="DC (1, 0)", DD="DD (0, 0)")
    legend_plotted = list()

    bar_locs = np.array([-3, -1, 1, 3]) * 0.1
    bar_width = 0.1
    from scipy.stats import gaussian_kde
    kdes = dict()
    maxval = 0
    for i_emotion, emotion in enumerate(emotions):
        # print(f"make kde for {i_emotion + 1}")
        kdes[emotion] = dict()
        for i_outcome, outcome in enumerate(outcomes):
            observations = ppldataemodf[outcome].loc[:, ('emotionIntensities', emotion)]

            support_ = np.arange(0, 1.01, 0.01)
            kde = gaussian_kde(observations, bw_method=bandwidth)
            pd_ = kde.pdf(support_)
            kdes[emotion][outcome] = pd_ / np.sum(pd_)

    for i_emotion, emotion in enumerate(emotions):
        for i_outcome, outcome in enumerate(outcomes):
            i_x_major = i_emotion
            i_x_minor = i_outcome

            bar_loc_center = i_x_major + bar_locs[i_x_minor]
            bar_loc_l = bar_loc_center - (bar_width / 2)
            bar_loc_r = bar_loc_center + (bar_width / 2)

            kkk = scale_factor * kdes[emotion][outcome]  # / np.max(kdes[emotion][outcome])

            if outcome == 'CC':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'CD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DC':
                cdict3 = {'red': [[0.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }

            X_ = np.tile(kkk, [2, 1]).T
            cutsom_cmap3 = LinearSegmentedColormap('testCmap', segmentdata=cdict3, N=256)
            gradient_image(ax, X_, extent=(bar_loc_l, bar_loc_r, 0.0, 1.0), direction=0, cmap=cutsom_cmap3, cmap_range=(0, 1.0), aspect='auto', zorder=50)

            # scatter_nudge = 0.02
            scatter_nudge = 0.0
            if outcome in legend_plotted:
                legend_label = None
            else:
                legend_label = legend_str[outcome]
                legend_plotted.append(outcome)
            color = dict(CC='green', CD='blue', DC='red', DD='black')[outcome]

            ax.scatter([bar_loc_center + scatter_nudge], [emoevdf.loc[outcome, emotion]], marker='o', s=45, color=color, facecolor='white', linewidth=1.5, zorder=51, label=legend_label)

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(np.arange(0, 20))
    if emotions_abbriv is None:
        ax.set_xticklabels(emotions, rotation=-35, rotation_mode='anchor', ha='left')
    else:
        emotions_abbriv_labels = [emotions_abbriv[emotion] for emotion in emotions]
        ax.set_xticklabels(emotions_abbriv_labels)

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='both', width=0, length=0)  # , labelsize=11

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    # ax.grid(visible=None, which='major', axis='both', **kwargs)
    ax.xaxis.grid(visible=True, which='minor')
    ax.xaxis.grid(visible=False, which='major')

    ax.legend(loc='lower right', bbox_to_anchor=(0.98, 1.01), ncol=4, frameon=False, fontsize=11, handlelength=0.9, handletextpad=0.3)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    plt.close('all')

    # %%


def new_emo_plot_iaf(ppldataemodf, emoevdf=None, scale_factor=1.0, bandwidth=0.05, emotions_abbriv=None, fig_outpath=None, plotParam=None):

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    """
    e.g.
    ppldataemodf = ppldata['empiricalEmotionJudgments']
    """
    # %%

    def gradient_image(ax, X, extent, direction=0.0, cmap_range=(0, 1), **kwargs):
        """
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* keyword argument.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.
        """

        im = ax.imshow(np.flipud(X), extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, **kwargs)
        return im

    plt = plotParam['plt']

    outcomes = plotParam['outcomes']

    emotions = ppldataemodf['CC'].loc[:, 'emotionIntensities'].columns.tolist()

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(9.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(2.0), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    legend_str = dict(CC="CC (0.5, 0.5)", CD="CD (0, 1)", DC="DC (1, 0)", DD="DD (0, 0)")
    legend_plotted = list()

    bar_locs = np.array([-3, -1, 1, 3]) * 0.1
    bar_width = 0.1
    from scipy.stats import gaussian_kde
    kdes = dict()
    maxval = 0
    for i_emotion, emotion in enumerate(emotions):
        # print(f"make kde for {i_emotion + 1}")
        kdes[emotion] = dict()
        for i_outcome, outcome in enumerate(outcomes):
            observations = ppldataemodf[outcome].loc[:, ('emotionIntensities', emotion)]

            support_ = np.arange(-2, 2.01, 0.01)
            try:
                kde = gaussian_kde(observations, bw_method=bandwidth)
                pd_ = kde.pdf(support_)
                kdes[emotion][outcome] = pd_ / np.sum(pd_)
            except:
                kdes[emotion][outcome] = np.zeros(len(support_))

    for i_emotion, emotion in enumerate(emotions):
        for i_outcome, outcome in enumerate(outcomes):
            i_x_major = i_emotion
            i_x_minor = i_outcome

            bar_loc_center = i_x_major + bar_locs[i_x_minor]
            bar_loc_l = bar_loc_center - (bar_width / 2)
            bar_loc_r = bar_loc_center + (bar_width / 2)

            kkk = scale_factor * kdes[emotion][outcome]  # / np.max(kdes[emotion][outcome])

            if outcome == 'CC':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'CD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DC':
                cdict3 = {'red': [[0.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }

            X_ = np.tile(kkk, [2, 1]).T
            cutsom_cmap3 = LinearSegmentedColormap('testCmap', segmentdata=cdict3, N=256)
            gradient_image(ax, X_, extent=(bar_loc_l, bar_loc_r, -2.0, 2.0), direction=0, cmap=cutsom_cmap3, cmap_range=(0, 1.0), aspect='auto', zorder=50)

            # scatter_nudge = 0.02
            scatter_nudge = 0.0
            if outcome in legend_plotted:
                legend_label = None
            else:
                legend_label = legend_str[outcome]
                legend_plotted.append(outcome)
            color = dict(CC='green', CD='blue', DC='red', DD='black')[outcome]

            ax.scatter([bar_loc_center + scatter_nudge], [emoevdf.loc[outcome, emotion]], marker='o', s=45, color=color, facecolor='white', linewidth=1.5, zorder=51, label=legend_label)

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(np.arange(0, len(emotions)))
    if emotions_abbriv is None:
        ax.set_xticklabels(emotions, rotation=-35, rotation_mode='anchor', ha='left')
    else:
        emotions_abbriv_labels = [emotions_abbriv[emotion] for emotion in emotions]
        ax.set_xticklabels(emotions_abbriv_labels, rotation=-35, rotation_mode='anchor', ha='left')

    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='both', width=0, length=0)  # , labelsize=11

    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    # ax.grid(visible=None, which='major', axis='both', **kwargs)
    ax.xaxis.grid(visible=True, which='minor')
    ax.xaxis.grid(visible=False, which='major')

    ax.legend(loc='lower right', bbox_to_anchor=(0.98, 1.01), ncol=4, frameon=False, fontsize=11, handlelength=0.9, handletextpad=0.3)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    plt.close('all')

    # %%


def get_iaf_names(cfg):
    import pickle

    from webpypl_emotionfunction_crossvalidation import prep_generic_data_pair_, prep_specific_data_pair_
    from webpypl import getEmpiricalModel_pair_

    cpar = cfg['cpar']
    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    with open(cpar.paths['wpplDataCache'], 'rb') as f:
        ppldata, ppldata_exp3, distal_prior_ppldata, wpplparam = pickle.load(f)

    feature_selector_label, feature_selector = cpar.pytorch_spec['feature_selector']

    composite_emodict_, composite_iafdict_ = prep_generic_data_pair_(ppldata)
    Y_full, X_full, _ = getEmpiricalModel_pair_(composite_emodict_, composite_iafdict_, feature_selector=feature_selector, return_ev=False)

    return X_full.drop(['pot', 'outcome'], axis=1).columns.to_list()


def format_iaf_labels_superchar(iaf_labels):

    iaf_labels_formatted = list()
    for label in iaf_labels:
        label1 = label.replace('U[', 'AU[').replace('a1[', '_{a1}[').replace('a2[', '_{a2}[').replace('[base', '^{~\mathbf{b}~}[').replace('[repu', '^{~\mathbf{r}~}[').replace('PEa2lnpot', 'PE[a_2]')
        labelf = r"$" + label1 + r"$"
        iaf_labels_formatted.append(labelf)

    return iaf_labels_formatted


def format_iaf_labels_superphrase(iaf_labels):

    iaf_labels_formatted = list()
    for label in iaf_labels:
        label1 = label.replace('U[', 'AU[').replace('a1[', '_{a1}[').replace('a2[', '_{a2}[').replace('[base', '^{~\mathbf{base}~}[').replace('[repu', '^{~\mathbf{repu}~}[').replace('PEa2lnpot', 'PE[a_2{=}\mathrm{C}]')
        labelf = r"$" + label1 + r"$"
        iaf_labels_formatted.append(labelf)

    return iaf_labels_formatted


def format_iaf_labels_inlinechar(iaf_labels):

    iaf_labels_formatted = list()
    for label in iaf_labels:
        label1 = label.replace('U[', 'AU[').replace('a1[', '_{a1}[').replace('a2[', '_{a2}[').replace('[base', '\,[\,\mathbf{b},\,').replace('[repu', '\,[\,\mathbf{r},\,').replace('PEa2lnpot', 'PE[a_2]')
        labelf = r"$" + label1 + r"$"
        iaf_labels_formatted.append(labelf)

    return iaf_labels_formatted


def composite_Aweights(learned_param_Amean=None, emo_labels=None, iaf_labels=None, fig_outpath=None, plotParam=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    sns = plotParam['sns']
    #######

    plt.close('all')

    no_debug = True

    text_size = 7

    iaf_labels_formatted = format_iaf_labels_superphrase(iaf_labels)

    A_ = learned_param_Amean.T
    maxval = np.max(np.absolute(A_.flatten()))

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(4.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(3.75), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    # fig, ax = plt.subplots(figsize=(12, 10))

    cmap_ = sns.diverging_palette(260, 12, s=99, l=40, sep=2, as_cmap=True)
    ax = sns.heatmap(A_, annot=True, linewidths=.5, center=0, vmin=-1 * maxval, vmax=maxval, cmap=cmap_, annot_kws={"size": 5}, fmt=".1f", ax=ax, cbar=False)
    ax.set_xticklabels(emo_labels, rotation=-35, horizontalalignment='left', rotation_mode='anchor', fontdict={'fontsize': text_size})
    ax.set_yticklabels(iaf_labels_formatted, rotation=0, horizontalalignment='right', fontdict={'fontsize': text_size})
    # ax.set_title(f"{mname} -- {set_id}")

    ax.tick_params(axis="both", pad=-3, labelsize=text_size)

    ax.set_ylim((A_.shape[0], 0))

    # %%
    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def single_Aweights(learned_param_A=None, emotion=None, emo_labels=None, iaf_labels=None, xrange=None, cialpha=None, fig_outpath=None, plotParam=None):

    import numpy as np
    import pandas as pd
    # %%

    plt = plotParam['plt']
    sns = plotParam['sns']
    #######

    plt.close('all')

    no_debug = True

    i_emo = emo_labels.index(emotion)

    plt = plotParam['plt']
    emoparam = pd.DataFrame(learned_param_A[:, i_emo, :], columns=iaf_labels)

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(1.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(3.75), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    for i_feature, feature in enumerate(iaf_labels):
        # from scipy.stats import norm
        # alpha = .05
        # l, u = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
        xxx = emoparam.loc[:, feature].to_list()
        alphatrim = len(xxx) * (cialpha / 2)
        alphatrim_int = int(alphatrim)
        xxxci = sorted(xxx)[alphatrim_int:-alphatrim_int]
        ci_ = [np.min(xxxci), np.max(xxxci)]
        # ax.errorbar([np.mean(xxx)], [i_feature], xerr=np.array([ci_]).T, color='black', alpha=0.95, capsize=2, elinewidth=0.5)
        if np.min(xxxci) <= 0 and np.max(xxxci) >= 0:
            color_ = 'dimgrey'
        else:
            if np.min(xxxci) > 0:
                color_ = 'firebrick'
            elif np.max(xxxci) < 0:
                color_ = 'royalblue'
            else:
                color_ = 'green'
        ax.scatter([np.mean(xxx)], [i_feature], color=color_, s=10, alpha=1.0)
        ax.plot([np.min(xxxci), np.max(xxxci)], [i_feature, i_feature], color=color_, alpha=0.8, linewidth=1)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)
    ax.set_yticks(range(len(iaf_labels)))
    iaf_labels_formatted = format_iaf_labels_superchar(iaf_labels)
    ax.set_yticklabels(iaf_labels_formatted)
    ax.set_xscale('symlog', base=np.e)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    # ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4, 5])
    xlim = ax.get_xlim()
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 1))
    if xrange is not None:
        ax.set_xlim(xrange)
    # ax.set_xlim([-2, 5])
    ax.set_xticklabels([])

    ax.set_ylim((18.5, -0.5))

    text_size = 7
    ax.tick_params(axis="both", pad=-3, labelsize=text_size)

    ### For axis to appear in the same pixel location, regardless of labels,
    # plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    # plt.close('all')


def composite_emos_summarybars(bardatadf, x_tick_order=None, model_colors=None, model_order=None, fig_outpath=None, plotParam=None, x_tick_labels=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    # {'fontsize':11*scale_temp,'horizontalalignment':'left'}, rotation=-35, rotation_mode='anchor'
    text_size_small = 10
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'center'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ##############
    ### TRAIN ON GENERIC, TEST ON DISTAL DELTAS, BARS AS CORRELATIONS #### NEW CORRELATION
    #### NEW CORRELATION
    ##############

    ####################
    # train on generic, test on distal
    # plots bars of correlation for each stimulus with 95% ci
    ####################

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}

    ax = fig.add_subplot(gridspec[gs_row['players'], gs_col['fits']])
    axs.append(ax)

    ax = plotGroupedBars_new_withci_resized(bardatadf, ax=ax, colorin=model_colors, xlabels=x_tick_order, modellabels=model_order)
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, 0.5])
    ax.xaxis.grid(False)
    ax.set_xticklabels(['Overall'], fontdict=fontdict_ticks)
    ax.tick_params(axis="x", labelsize=text_size_large, pad=0)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def composite_legend(model_labels=None, model_colors=None, fig_outpath=None, plotParam=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    # {'fontsize':11*scale_temp,'horizontalalignment':'left'}, rotation=-35, rotation_mode='anchor'
    text_size_small = 10
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ax00 = fig.add_subplot(gridspec[gs_row['mt'], gs_col['emo']])

    legend_elements = [
        Patch(label=model_labels['iaaFull'], facecolor=model_colors['iaaFull'], edgecolor='none', linewidth=0.),
        Patch(label=model_labels['invplanLesion'], facecolor=model_colors['invplanLesion'], edgecolor='none', linewidth=0.),
        Patch(label=model_labels['socialLesion'], facecolor=model_colors['socialLesion'][0], edgecolor='none', linewidth=0.),
        Patch(label='Empirical CI', facecolor=(0., 0., 0., 0.2), edgecolor=(0., 0., 0., 0.6), linewidth=0.5, linestyle='-'),  # edgecolor=(1., 0., 0., 0.4), linestyle=(0, (3,1))
    ]

    ax00.set_axis_off()
    ax00.legend(handles=legend_elements, loc='center', columnspacing=1.0, bbox_to_anchor=(0.5, 0.8), frameon=False, handletextpad=0.3, ncol=len(legend_elements), prop={'size': text_size_small})  # bbox_to_anchor=(0.5, -0.3) ### , title="title"

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def composite_emos_concord(bardatadf, x_tick_order=None, model_colors=None, model_order=None, fig_outpath=None, plotParam=None, x_tick_labels=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    # {'fontsize':11*scale_temp,'horizontalalignment':'left'}, rotation=-35, rotation_mode='anchor'
    text_size_small = 10
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ##############
    ### TRAIN ON GENERIC, TEST ON DISTAL DELTAS, BARS AS CORRELATIONS #### NEW CORRELATION
    #### NEW CORRELATION
    ##############

    ####################
    # train on generic, test on distal
    # plots bars of correlation for each stimulus with 95% ci
    ####################

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}

    ax = fig.add_subplot(gridspec[gs_row['players'], gs_col['emo']])
    axs.append(ax)

    ax = plotGroupedBars_new_withci_resized(bardatadf, ax=ax, colorin=model_colors, xlabels=x_tick_order, modellabels=model_order)
    ylim_original = ax.get_ylim()

    ax.set_ylabel(ax.get_ylabel(), fontdict=fontdict_axislab)

    ax.xaxis.grid(False)
    ax.set_xlim((-0.75, 19.75))
    # ax.set_ylim((ax.get_ylim()[0],0.4))

    # ax.set_ylim([-0.2, 1.0])
    # ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if x_tick_labels is None:
        x_labels = x_tick_order
    else:
        x_labels = [x_tick_labels[stimid] for stimid in x_tick_order]
    # ax = axs[-1]

    ax.tick_params(axis="x", labelsize=text_size_large, pad=0)
    ax.tick_params(axis="y", labelsize=text_size_small, pad=0)

    aaa = ax.set_xticklabels(x_labels, rotation=-35, horizontalalignment='left', rotation_mode='anchor', fontdict={'fontsize': text_size_large})

    ylim_ = (np.sign(ylim_original[0]) * np.ceil(abs(ylim_original[0] * 10)) / 10, np.sign(ylim_original[1]) * np.ceil(abs(ylim_original[1] * 10)) / 10)
    ax.set_ylim(ylim_)

    ycoord = ylim_[0] - ((ylim_[1] - ylim_[0]) * 0.07)

    exceed_ax = False
    for xcord in np.arange(19) + 0.5:
        ax.plot([xcord, xcord], [ycoord, ylim_[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ax.set_ylim(ylim_)

    i_ax = 1
    exceed_ax = False
    if i_ax == 1:
        from math import pi
        inv = ax.transData.inverted()
        magnitude = 40
        angle = -35. * (pi / 180.)
        dxx = magnitude * np.cos(angle)
        dyy = magnitude * np.sin(angle)
        p0d = ax.transData.transform([0, ycoord])
        # exceed_ax = False
        for xcord in np.arange(19) + 0.5:

            p1d = ax.transData.transform([xcord, ycoord])
            p2d = [p1d[0] + dxx, p0d[1] + dyy]
            p2cord = inv.transform(p2d)

            ax.plot([xcord, p2cord[0]], [ycoord, p2cord[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ########################

    fig.align_ylabels(axs)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def composite_emos_concord_iaaonly(bardatadf, x_tick_order=None, model_colors=None, model_order=None, fig_outpath=None, plotParam=None, x_tick_labels=None, ylim_fixed=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    # {'fontsize':11*scale_temp,'horizontalalignment':'left'}, rotation=-35, rotation_mode='anchor'
    text_size_small = 10
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ##############
    ### TRAIN ON GENERIC, TEST ON DISTAL DELTAS, BARS AS CORRELATIONS #### NEW CORRELATION
    #### NEW CORRELATION
    ##############

    ####################
    # train on generic, test on distal
    # plots bars of correlation for each stimulus with 95% ci
    ####################

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}

    ax = fig.add_subplot(gridspec[gs_row['players'], gs_col['emo']])
    axs.append(ax)

    ax = plotGroupedBars_new_withci_resized_iaaonly(bardatadf, ax=ax, colorin=model_colors, xlabels=x_tick_order, modellabels=model_order)
    ylim_original = ax.get_ylim()

    ax.set_ylabel(ax.get_ylabel(), fontdict=fontdict_axislab)

    ax.xaxis.grid(False)
    ax.set_xlim((-0.75, 19.75))
    # ax.set_ylim((ax.get_ylim()[0],0.4))

    # ax.set_ylim([-0.2, 1.0])
    # ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    if x_tick_labels is None:
        x_labels = x_tick_order
    else:
        x_labels = [x_tick_labels[stimid] for stimid in x_tick_order]
    # ax = axs[-1]

    ax.tick_params(axis="x", labelsize=text_size_large, pad=0)
    ax.tick_params(axis="y", labelsize=text_size_small, pad=0)

    aaa = ax.set_xticklabels(x_labels, rotation=-35, horizontalalignment='left', rotation_mode='anchor', fontdict={'fontsize': text_size_large})

    ylim_ = (np.sign(ylim_original[0]) * np.ceil(abs(ylim_original[0] * 10)) / 10, np.sign(ylim_original[1]) * np.ceil(abs(ylim_original[1] * 10)) / 10)

    if ylim_fixed is not None:
        ylim_ = ylim_fixed

    ax.set_ylim(ylim_)

    ycoord = ylim_[0] - ((ylim_[1] - ylim_[0]) * 0.07)

    exceed_ax = False
    for xcord in np.arange(19) + 0.5:
        ax.plot([xcord, xcord], [ycoord, ylim_[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ax.set_ylim(ylim_)

    i_ax = 1
    exceed_ax = False
    if i_ax == 1:
        from math import pi
        inv = ax.transData.inverted()
        magnitude = 40
        angle = -35. * (pi / 180.)
        dxx = magnitude * np.cos(angle)
        dyy = magnitude * np.sin(angle)
        p0d = ax.transData.transform([0, ycoord])
        # exceed_ax = False
        for xcord in np.arange(19) + 0.5:

            p1d = ax.transData.transform([xcord, ycoord])
            p2d = [p1d[0] + dxx, p0d[1] + dyy]
            p2cord = inv.transform(p2d)

            ax.plot([xcord, p2cord[0]], [ycoord, p2cord[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ########################

    fig.align_ylabels(axs)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def plotGroupedBars_new_withci_resized(bardatadf, ax=None, colorin=None, xlabels=None, modellabels=None, width=0.35, major_space=0.35, minor_space=0.04):
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    n_major = len(xlabels)
    n_minor = len(modellabels)

    width = 0.23
    minor_space = 0.04
    ### 1 unit = (3 bars) + (2 minor space) + (1 major space)

    adjuster = [-1 * (width + minor_space), 0, (width + minor_space)]

    errorboxes = list()
    for i_x, xlabel in enumerate(xlabels):
        for i_y, model in enumerate(modellabels):
            ydata_ = bardatadf.loc[(bardatadf['model'] == model) & (bardatadf['xlabel'] == xlabel), :]
            y_ = ydata_['pe']
            ax.bar(i_x + adjuster[i_y], y_, width, color=colorin[model], edgecolor='none')

            cil_ = ydata_['cil'].item()
            ciu_ = ydata_['ciu'].item()
            ax.plot((i_x + adjuster[i_y], i_x + adjuster[i_y]), [cil_, ciu_], color='k', linewidth=1)

    for i_x, xlabel in enumerate(xlabels):
        ydata_ = bardatadf.loc[(bardatadf['model'] == 'empirical') & (bardatadf['xlabel'] == xlabel), :]
        cil_ = ydata_['cil'].item()
        ciu_ = ydata_['ciu'].item()
        # Loop over data points; create box from errors at each point
        empci_x_buffer = minor_space
        ci_emp_width = (width * len(adjuster)) + (minor_space * (len(adjuster) - 1)) + (empci_x_buffer * 2)
        x_emp_left = i_x - ci_emp_width / 2
        rect = Rectangle((x_emp_left, cil_), ci_emp_width, ciu_ - cil_)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pce = PatchCollection(errorboxes, facecolor='none', alpha=0.6, edgecolor='k', linewidths=0.5, zorder=11)
    _ = ax.add_collection(pce)

    pc = PatchCollection(errorboxes, facecolor='k', alpha=0.2, edgecolor='none', zorder=10)
    _ = ax.add_collection(pc)

    _ = ax.set(xticks=range(n_major), xticklabels=xlabels)

    ax.set_ylim([bardatadf.loc[:, 'cil'].min(), bardatadf.loc[:, 'ciu'].max()])

    return ax


def plotGroupedBars_new_withci_resized_iaaonly(bardatadf, ax=None, colorin=None, xlabels=None, modellabels=None, width=0.35, major_space=0.35, minor_space=0.04):
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    n_major = len(xlabels)
    n_minor = len(modellabels)

    width = 0.23
    minor_space = 0.04
    ### 1 unit = (3 bars) + (2 minor space) + (1 major space)

    adjuster = [-1 * (width + minor_space), 0, (width + minor_space)]

    errorboxes = list()
    for i_x, xlabel in enumerate(xlabels):
        for i_y, model in enumerate(modellabels):
            ydata_ = bardatadf.loc[(bardatadf['model'] == model) & (bardatadf['xlabel'] == xlabel), :]
            y_ = ydata_['pe']
            ax.bar(i_x + adjuster[i_y], y_, width, color=colorin[model], edgecolor='none')

            if colorin[model] == '#00000000':
                errbarcolor = '#00000000'
            else:
                errbarcolor = 'k'
            cil_ = ydata_['cil'].item()
            ciu_ = ydata_['ciu'].item()
            ax.plot((i_x + adjuster[i_y], i_x + adjuster[i_y]), [cil_, ciu_], color=errbarcolor, linewidth=1)

    for i_x, xlabel in enumerate(xlabels):
        ydata_ = bardatadf.loc[(bardatadf['model'] == 'empirical') & (bardatadf['xlabel'] == xlabel), :]
        cil_ = ydata_['cil'].item()
        ciu_ = ydata_['ciu'].item()
        # Loop over data points; create box from errors at each point
        empci_x_buffer = minor_space
        ci_emp_width = (width * len(adjuster)) + (minor_space * (len(adjuster) - 1)) + (empci_x_buffer * 2)
        x_emp_left = i_x - ci_emp_width / 2
        rect = Rectangle((x_emp_left, cil_), ci_emp_width, ciu_ - cil_)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pce = PatchCollection(errorboxes, facecolor='none', alpha=0.6, edgecolor='k', linewidths=0.5, zorder=11)
    _ = ax.add_collection(pce)

    pc = PatchCollection(errorboxes, facecolor='k', alpha=0.2, edgecolor='none', zorder=10)
    _ = ax.add_collection(pc)

    _ = ax.set(xticks=range(n_major), xticklabels=xlabels)

    ax.set_ylim([bardatadf.loc[:, 'cil'].min(), bardatadf.loc[:, 'ciu'].max()])

    return ax


def plot_webppl_followup_results(cpar_path_full, cpar_path_priors, outpath_base):
    from react_collect_pytorch_cvresults import get_ppldata_cpar
    from webpypl_plotfun import init_plot_objects
    from iaa21_runwebppl_followup import followup_analyses
    from react_collect_pytorch_cvresults import plot_webppl_data
    import dill

    plotParam = init_plot_objects(outpath_base)

    with open(cpar_path_priors, 'rb') as f:
        cpar_priors = dill.load(f)
    cpar_priors.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    ppldata_priors, _, _ = get_ppldata_cpar(cpar_priors)
    print(f"\n\n-------followup_analyses cpar_priors")
    followup_analyses(cpar_priors, *get_ppldata_cpar(cpar_priors), plotParam=plotParam)
    print(f"\n\n-------plot_webppl_data cpar_priors")
    plot_webppl_data(cpar_path_priors, plotParam=plotParam)
    print(f"\n\n-------finished cpar_priors")

    with open(cpar_path_full, 'rb') as f:
        cpar = dill.load(f)
    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    ppldata, ppldata_exp3, distal_prior_ppldata = get_ppldata_cpar(cpar)

    print(f"\n\n-------followup_analyses cpar")
    followup_analyses(cpar, ppldata, ppldata_exp3, distal_prior_ppldata, plotParam=plotParam)
    print(f"\n\n-------plot_webppl_data cpar")
    plot_webppl_data(cpar_path_full, plotParam=plotParam)
    print(f"\n\n-------finished cpar")

    return 0
