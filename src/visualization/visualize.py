def residual_plot(model, x, y_true, sae, wkendall, meshblock):
    y_pred = model.predict(x)
    y_pred = pd.Series(y_pred, index=x.index, name='y_pred')
    
    rank_pred = pd.Series(rankdata(y_pred), index=x.index, name='rank_pred').astype('int64')
    rank_true = pd.Series(rankdata(y_true), index=x.index, name='rank_true').astype('int64')
    #OldRange = (y_pred.max() - y_pred.min())
    #NewRange = (y_true.max() - y_true.min())
    #y_pred = (((y_pred - y_pred.min()) * NewRange)/OldRange) + y_true.min()

    true_map = meshblock.merge(y_true, on='Cod_ap', how='left').copy()
    true_map = true_map.merge(rank_true, on='Cod_ap', how='left')
    true_map.dropna(axis=0, inplace=True)
    
    pred_map = meshblock.merge(y_pred, on='Cod_ap', how='left').copy()
    pred_map = pred_map.merge(rank_pred, on='Cod_ap', how='left')
    pred_map.dropna(axis=0, inplace=True)

    fig, ax = plt.subplots(2, 2)

    #ax[0][1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #ax[1][1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ax[0][0].set_xticks([]) 
    ax[0][0].set_yticks([])
    ax[0][1].set_xticks([]) 
    ax[0][1].set_yticks([]) 
    box = ax[0][1].get_position()
    ax[0][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    ax[0][0].set_title('True Distribution')
    ax[0][1].set_title('True Rank')
    true_map.plot(column='LISA', ax=ax[0][0], legend=True, cmap='RdYlBu')
    true_map.plot(column='rank_true', ax=ax[0][1], label='rank_pred', legend=True, cmap='RdYlBu')
    ax[1][0].set_xticks([]) 
    ax[1][0].set_yticks([]) 
    ax[1][1].set_xticks([]) 
    ax[1][1].set_yticks([]) 
    box = ax[1][1].get_position()
    ax[1][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    ax[1][0].set_title('Predicted Distribution')
    ax[1][1].set_title('Predicted Rank')
    
  
    # Text
    textstr = '\n'.join((
    r'$\mathrm{SAE}=%.2f$' % (sae, ),
    r'$Kendall=%.2f$' % (wkendall, )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)

    # place a text box in upper left in axes coords
    ax[1][0].text(0.05, 0.95, textstr, transform=ax[1][0].transAxes, fontsize=7,
        verticalalignment='top', bbox=props)   
    pred_map.plot(column='y_pred', ax=ax[1][0], legend=True, cmap='RdYlBu')
    
  
    pred_map.plot(column='rank_pred', ax=ax[1][1], label=pred_map['rank_pred'].values, legend=True, cmap='RdYlBu')

    
    
    
    #fig.tight_layout()
    pp.savefig()