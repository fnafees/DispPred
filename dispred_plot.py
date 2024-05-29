# Plot roc auc curve
def plot_roc_auc(y_test,y_proba,y_probaFl,run_name,sharedDir):
    # print("")
    print("\n","*"*20,"Plot roc auc curve","*"*20, "\n")
    # sns.set_palette("tab10") # "dark","bright","tab10"
    sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks

    fig= pyplot.figure(figsize=(5.5, 4),dpi=150)
    pyplot.rc('axes', titlesize=12)     # fontsize of the axes title
    pyplot.rc('axes', labelsize=11)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=11)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=11)    # fontsize of the tick labels
    pyplot.rc('legend', fontsize=9)    # legend fontsize
    pyplot.rc('font', size=11)          # controls default text sizes

    ns_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    ax = sns.lineplot(x=ns_fpr, y=ns_tpr,errorbar=None, linestyle='--') #, label='Random'

    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_proba)
    ax = sns.lineplot(x=lr_fpr, y=lr_tpr,errorbar=None, label='Proposed Method')

    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_probaFl)
    ax = sns.lineplot(x=lr_fpr, y=lr_tpr,errorbar=None,label='fIDPnn')

    df_allproba= pd.read_csv(sharedDir+"/files/AllProba.csv",index_col=0)
    df_allproba= df_allproba[["LEspritzProba","LIUPRED2Proba", "SPOTProba" ,"netSufProba", "MetaProba"]]
    df_allproba.columns = ["Espritz","IUPred3", "SPOT-Disorder-Single", "NetSurfP - 2.0", "Metapredict" ]

    for name in df_allproba.columns:
        if name != "target":        
            lr_fpr, lr_tpr, _ = roc_curve(y_test, df_allproba[name].to_numpy())
            ax = sns.lineplot(x=lr_fpr, y=lr_tpr,errorbar=None,label=name)

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')

    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyplot.savefig("./output/"+run_name+"/images/ROCcurve.png",dpi=450, bbox_inches = 'tight')
    pyplot.show()


    mlflow.log_figure(fig,"./output/"+run_name+"/images/ROCcurve.png")
    pyplot.clf()
    
    
    # Plot Precision-Recall curve
def plot_precisionrecall(y_test,y_proba,y_probaFl,run_name,sharedDir):
    # print("")
    print("\n","*"*20,"Plot Precision-Recall curve","*"*20, "\n")
    fig= pyplot.figure(figsize=(5.5, 4),dpi=150)
    pyplot.rc('axes', titlesize=12)     # fontsize of the axes title
    pyplot.rc('axes', labelsize=11)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=11)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=11)    # fontsize of the tick labels
    pyplot.rc('legend', fontsize=9)    # legend fontsize
    pyplot.rc('font', size=11)          # controls default text sizes

    no_skill = len(y_test[y_test==1]) / len(y_test)
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_proba)
    ax = sns.lineplot(x=[0, 1], y=[no_skill, no_skill],errorbar=None,  linestyle='--') #label='Random',

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_proba)
    ax = sns.lineplot(x=lr_recall, y=lr_precision,errorbar=None, label='Proposed Method')
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probaFl)
    ax = sns.lineplot(x=lr_recall, y=lr_precision,errorbar=None,label='fIDPnn')

    df_allproba= pd.read_csv(sharedDir+"/files/AllProba.csv",index_col=0)
    df_allproba= df_allproba[["LEspritzProba","LIUPRED2Proba", "SPOTProba" ,"netSufProba", "MetaProba"]]
    df_allproba.columns = ["Espritz","IUPred3", "SPOT-Disorder-Single", "NetSurfP - 2.0", "Metapredict" ]

    for name in df_allproba.columns:
        if name != "target":        
            lr_precision, lr_recall, _ = precision_recall_curve(y_test, df_allproba[name].to_numpy())
            ax = sns.lineplot(x=lr_recall, y=lr_precision,errorbar=None,label=name)


    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')


    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyplot.savefig("./output/"+run_name+"/images/PRcurve.png",dpi=450, bbox_inches = 'tight')
    pyplot.show()
    mlflow.log_figure(fig,"./output/"+run_name+"/images/PRcurve.png")
    pyplot.clf()
    mlflow.end_run()
    
# # Comparison of the results with other methods
# def plot_comparisons(y_test,run_name):
#     print("Comparison of the results with other methods")
#     df2=pd.DataFrame()
#     df2['target'] = y_test.tolist()
#     # df2 = pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/5_Dispredict/test/MergeDataset/OriginalDatasetL01533/testFeaturesL01533.csv")
#     df1 = pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/script/netSuf/test_out/test.csv")

#     df_all=pd.DataFrame()
#     idlist=open("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/originalDataset/test_idList.txt", "r")

#     for id in idlist:
#         dff=pd.DataFrame()
#         # print(id[:-1])
        
#         df=pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/tools/Espritz/testFasta_long_0.26/"+id[:-1]+".espritz",skiprows=8,sep="\t",header=None)
#         df.columns=["A","Espritz"]
#         df["Espritz"]
#         dff=pd.concat([dff,df["Espritz"] ],axis=1)

#         df=pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/tools/iupred3/iupred3_long/"+id[:-1]+".txt",skiprows=12,sep="\t",header=None)
#         df.columns=["A","A","IUPred3","ANCHOR2"]
#         df["IUPred3"]
#         dff=pd.concat([dff,df["IUPred3"] ],axis=1)

#         df=pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/tools/SPOT-Disorder-Single/test_output/"+id[:-1]+".spotds",skiprows=2,sep="\t",header=None)
#         df.columns=["A","B","SPOT-Disorder-Single","D"]
#         df["SPOT-Disorder-Single"]
#         dff=pd.concat([dff,df["SPOT-Disorder-Single"] ],axis=1)
       
#         df3=df1[df1["id"]==id[:-1]]["disorder"].reset_index()
#         df3.columns=["A","NetSurfP - 2.0"]
#         df3["NetSurfP - 2.0"]
#         dff=pd.concat([dff,df3["NetSurfP - 2.0"] ],axis=1)
#         # print(dff)
#         df_all=pd.concat([df_all,dff ])
#     df_all.reset_index(inplace=True,drop=True)
#     df_all=pd.concat([df_all,df2["target"] ],axis=1)
#     df_all

#     df=pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/6_DispredictProposal/script/MetaPredict/test.csv",index_col=0)
#     df.reset_index(inplace=True,drop=True)
#     df_all["Metapredict"]=df["MetaProba"]
    
#     df=pd.read_csv("/home/mkabir3/Research/13_Dispredict_Restart/5_Dispredict/old3/test_proba.csv",index_col=0)
#     df_all["DisPredict3.0"]=df.iloc[:,1]

#     df_final_res=pd.DataFrame(columns=["Method", "AUC", "F1-score", "MCC","Kappa"])
#     r=0
#     for name in df_all.columns:
#         if name != "target":
#             res=classificationScore2(name, df_all["target"].to_numpy() ,df_all[name].to_numpy(),threshold=0.5)
#             # print(res)
#             df_final_res.loc[r,:]=[name]+ res
#             r=r+1
#     df_final_res

#     df1_transposed = df_final_res  
#     df1_transposed = df1_transposed.T
#     df1_transposed.columns=df1_transposed.loc["Method"]
#     df1_transposed=df1_transposed[1:]
#     df1_transposed["Metrics"]=df1_transposed.index

#     sns.set_style("whitegrid")
#     pyplot.rc('axes', titlesize=12)     # fontsize of the axes title
#     pyplot.rc('axes', labelsize=11)    # fontsize of the x and y labels
#     pyplot.rc('xtick', labelsize=11)    # fontsize of the tick labels
#     pyplot.rc('ytick', labelsize=11)    # fontsize of the tick labels
#     pyplot.rc('legend', fontsize=9)    # legend fontsize
#     pyplot.rc('font', size=11)          # controls default text sizes

#     fig, ax = pyplot.subplots()
#     sns.barplot(data=df1_transposed.melt(id_vars='Metrics',
#                                     value_name='Score', var_name='Methods'),
#                                         x='Metrics', y='Score', hue='Methods')

#     fig.set_size_inches(8,5)
#     # fig.set_dpi(350) 
#     pyplot.legend(loc='upper right', bbox_to_anchor=( 1,1))
#     pyplot.savefig("./output/"+run_name+"/images/ComparisonWOtherMethods.png",dpi=450, bbox_inches='tight') #, bbox_inches='tight'
#     pyplot.show()
#     pyplot.clf()