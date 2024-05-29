
import pandas as pd
import shap
import numpy as np

def shapImportance(model, X, y,run_name):
    explainer = shap.Explainer(model[-1])
    shap_values = explainer(X)
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(columns=['feature_name', 'shap_importance'])
    importance_df['feature_name']= X.columns
    importance_df['shap_importance'] = shap_sum
    importance_df = importance_df.sort_values('shap_importance', ascending=False)

    feature_list = importance_df[importance_df.shap_importance > 0.1]['feature_name'].tolist()
    importance_df.to_csv("./output_selectedfeat/"+run_name+"/shap_importance.csv")
    # feature_list = importance_df['features'].head(5).tolist()

    print('Number of features selected:', len(feature_list))
    print('Selected features:', feature_list)  
    return feature_list





